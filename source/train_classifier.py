""" 
Script for training text classification head given a pre-trained embedding model
"""

import os
import re
import json
import logging
import pickle
from dataclasses import dataclass
from math import isclose
from typing import Annotated, Any, Literal, Type

import hydra
import lightning as L
import matplotlib.pyplot as plt
import torch
from langchain_core.embeddings import Embeddings
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, WrapSerializer, field_validator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset

from source import dataset, model
from source.dataset import EmailDataset, EmailElem, split_dataset
from source.model import EmbeddingsModel

logger = logging.getLogger(__name__)


@dataclass
class DataLoaders:
    """ Holder for pytorch train, validation, test dataloaders """

    train_dataloader: torch.utils.data.DataLoader
    val_dataloader: torch.utils.data.DataLoader
    test_dataloader: torch.utils.data.DataLoader


@dataclass
class Datasets:
    """ Holder for pytorch train, validation, test datasets objects """

    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset


def class_serializer(value: Any, handler, info) -> str:
    """ Custom class json serializer """

    if info.mode == "json":
        return value.__name__

    return handler(value, info) 


class TrainCfg(BaseModel):
    """ Holder for training parameters """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    classification_head: Literal["deep", "logistic"]
    """ Whether to train deep or logistic regression head """

    dataset_class: Annotated[Type[Dataset], WrapSerializer(class_serializer)]
    model_class: Annotated[Type[Embeddings], WrapSerializer(class_serializer)]
    pretrained_model_dir: str
    """ Directory with the pre-trained embedding model checkpoints and config files """

    dataset_dir: str
    """ The directory with the dataset """

    train_ratio: float
    """  
    train_ratio: the ratio of the dataset to be used for training

    train_ratio + val_ratio + test_ratio has to sum to 1 
    """

    val_ratio: float
    """  
    val_ratio: the ratio of the dataset to be used for hyperparameters tuning.

    train_ratio + val_ratio + test_ratio has to sum to 1 
    """

    test_ratio: float
    """  
    test_ratio: the ratio of the dataset to be used for testing

    train_ratio + val_ratio + test_ratio has to sum to 1 
    """

    batch_size: int
    loss_fn: Annotated[Type[_Loss], WrapSerializer(class_serializer)]
    num_epochs: int
    output_experiments_directory: str
    """ The directory for saving experiments settings and results """

    seed: int

    @field_validator("dataset_class", mode="before")
    @classmethod
    def dataset_class_validator(cls, value: str) -> Type[Dataset]:
        """ Create and validate the class type """

        dataset_cls = getattr(dataset, value)
        return dataset_cls
    

    @field_validator("model_class", mode="before")
    @classmethod
    def model_class_validator(cls, value: str) -> Type[Embeddings]:
        """ Create and validate the model type """

        model_cls = getattr(model, value)
        return model_cls
    

    @field_validator("loss_fn", mode="before")
    @classmethod
    def loss_fn_validator(cls, value: str) -> Type[_Loss]:
        """ Create and validate the loss function type """

        loss_fn = getattr(torch.nn, value)
        return loss_fn
    
    
    def model_post_init(self, __context):
        assert isclose(
            self.train_ratio + self.val_ratio + self.test_ratio, 1.0
        ), f"the self.train_ratio + self.val_ratio + self.test_ratio = {self.train_ratio + self.val_ratio + self.test_ratio} is not close to 1.0."


def set_experiment_dir(output_directory: str, cfg: TrainCfg) -> tuple[str, int]:
    """ Create a new experiment directory for storing experiment config, model checkpoints, monitoring data and evaluation results 
    
    Args:
        output_directory: the directory for saving experiments related data

    Returns:
        the experiment directory
        the experiment id
    """

    if not os.path.exists(output_directory):
        os.mkdir(path=output_directory )
    assert os.path.isdir(output_directory)

    prev_dirs = [
        x
        for x in os.listdir(output_directory)
        if os.path.isdir(os.path.join(output_directory, x))
    ]

    prev_run: list[re.Match[str] | None] = [re.match(r"^\d+", x) for x in prev_dirs]
    
    prev_run_ids = [int(x.group()) for x in prev_run if x is not None]
    run_id = max(prev_run_ids, default=-1) + 1


    dir_desc = f"{cfg.classification_head}"
    run_dir = os.path.join(output_directory, f"{run_id:05d}-{dir_desc}")
    assert not os.path.exists(run_dir)
    os.mkdir(run_dir)
    assert os.path.exists(run_dir)


    OmegaConf.save(
        config=OmegaConf.create(cfg.model_dump(mode="json"))
        f=os.path.join(run_dir, "config.yaml")
    )

    return run_dir, run_id


def get_text_and_label(dataset: EmailDataset | list[EmailElem]) -> tuple[list[str], list[str]]:
    """ 
     Extract text and it's label from the dataset.add()

     Args:
        dataset: the dataset

    Returns:
        list of loaded texts
        list of loaded labels
    """

    text, label = [], []
    for elem in dataset:
        text.append(elem.text)
        label.append(elem.label)

    return text, label


class EmailDatasetWrapper(EmailDataset):
    """ 
    Create pytorch dataset that yields tensors as an output. Not EmailElem
    """

    def __init__(self, dataset_path = ..., cache = True, extention = "msg", limit_number_of_samples = 20):
        super().__init__(dataset_path, cache, extention, limit_number_of_samples)


    @classmethod
    def from_email(cls: "EmailDatasetWrapper", email_dataset: EmailDataset) -> "EmailDatasetWrapper":
        new_dataset = EmailDatasetWrapper(None)
        new_dataset.dict_data = email_dataset.dict_data
        new_dataset.all_files = email_dataset._all_files
        new_dataset.data = email_dataset.data
        new_dataset.email_classes = email_dataset.email_classes

        for idx, (key, value) in enumerate(new_dataset.email_classes.items()):
            new_dataset.email_classes[key] = idx
        
        return new_dataset


    def __getitem__(self, index):
        elem = super().__getitem__(index)
        return elem.email_text, self.email_classes[elem.email_cls]


class ClassificationModel(nn.Module):

    def __init__(
            self,
            embedding_cls: Type[nn.Module],
            classification_cls: Type[nn.Module],
            embedding_checkpoint: str,
            classification_checkpoint: str,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedding_model = embedding_cls(embedding_checkpoint)

        checkpoint = torch.load(classification_checkpoint)
        self.class_mappings = checkpoint["hyper_parameters"]["class_mapping"]
        self.inverted_class_mapping: dict[int, str] = {}
        for key, value in self.class_mappings.items():
            self.inverted_class_mapping[value] = key

        self.cls_head = classification_cls(
            input_dim = checkpoint["hyper_parameters"]["embedding_dim"],
            n_classes = len(list(self.class_mappings.keys())),
        )

        state_dict = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            state_dict[key.replace("cls_head."), ""] = state_dict.pop(key)

        self.cls_head.load_state_dict(state_dict)
        self.embedding_model.model.eval()
        self.cls_head.eval()

    
    @torch.inference_mode
    def forward(self, in_text) -> list[str]:
        embeddings = self.embedding_model(in_text)
        preds = self.cls_head(embeddings)
        cls_res = torch.argmax(nn.functional.softmax(preds, dim=-1), dim=-1)
        
        predicted_labels = []
        for pred_cls_idx in cls_res:
            predicted_labels.append(self.inverted_class_mapping[pred_cls_idx.item()])

        return predicted_labels
    

class DeepHead(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._input_dim = input_dim
        self._n_classes = n_classes
        self.cls_head = nn.Linear(
            in_features=self._input_dim, out_features=self._n_classes, bias=True
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        out = self.cls(embeddings)
        return out
    

class ClassificationTraining(L.LightningModule):
    """ Train the deep classification model """

    def __init__(
            self, 
            embeddings_model: type[Embeddings],
            embeddings_model_checkpoint: str,
            cls_head: Type[DeepHead],
            loss_fn: Type[_Loss],
            train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader,
            test_dataloader: torch.utils.data.DataLoader,
            learning_rate: float = 5e-5,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.embedding_model = embeddings_model(embeddings_model_checkpoint)
        self.embedding_model.model.eval()
        for param in self.embedding_model.model.parameters():
            param.requires_grad = False

        self.loss_fn = loss_fn()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        example_embedding = self.embedding_model(["dummy_text"])
        self.class_mapping = self.train_dataloader().dataset.email_classes
        self.classes = list(self.class_mapping.keys())
        self.n_classes = len(self.classes)
        self.embedding_dim = example_embedding.shape[-1]
        self.cls_head = cls_head(
            input_dim = example_embedding.shape[-1], n_classes=self.n_classes
        )
        self.cls_head.train()
        for param in self.cls_head.parameters():
            param.requires_grad = True

        self.save_hyperparameters(
            {
                "class_mapping": self.class_mapping,
                "embedding_dim": self.embedding_dim,
            },
            ignore=[
                "embeddings_model", 
                "cls_head",
                "loss_fn",
                "train_dataloader",
                "val_dataloader",
                "test_dataloader",
            ],
        )

    def forward(self, input_text: list[str], labels: list[str]):
        embeddings = self.embedding_model(input_text)
        preds = self.cls_head(embeddings.clone())
        loss = self.loss_fn(preds, labels)

        return loss, preds
    
    def training_step(self, batch, batch_idx):
        texts, labels = batch
        loss, preds = self(texts, labels)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        texts, labels = batch
        loss, preds = self(texts, labels)
        self.log("val_loss", loss)

        acc = accuracy_score(
            labels, torch.argmax(nn.functional.softmax(preds, dim=-1), dim=-1)
        )
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        texts, labels = batch
        loss, preds = self(texts, labels)
        self.log("test_loss", loss)

        acc = accuracy_score(
            labels, torch.argmax(nn.functional.softmax(preds, dim=-1), dim=-1)
        )
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.cls_head.parameters()),
            lr=self.learning_rate,
        )

        return optimizer

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader
    
    def test_dataloader(self):
        return self.test_dataloader
        
    
def create_dataloaders(
        train_d: EmailDataset,
        val_d: EmailDataset,
        test_d: EmailDataset,
        batch_size: int = 16,
) -> DataLoaders:
    """ 
    Create train, validation and test dataloaders

    Args:
        train_d: the training dataset
        val_d: the validation dataset
        test_d: the test dataset
        batch_size: the batch size

    Returns:
        train, validation and test dataloaders
    """
    train_dataloader = torch.utils.data.Dataloader(
        dataset=train_d,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    val_dataloader = torch.utils.data.Dataloader(
        dataset=val_d,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    test_dataloader = torch.utils.data.Dataloader(
        dataset=test_d,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    return DataLoaders(train_dataloader, val_dataloader, test_dataloader)
    

def train_logistic_head(
        datasets: Datasets,
        experiment_dir: str, 
        cfg: TrainCfg,
) -> tuple[dict[str, float], str]:
    """ 
    Train the logistic classification head

    Args:
        datasets: train, val and test datasets to train on
        experiment_dir: the experiments directory to save the training outcomes
        cfg: the training configurations

    Returns:
        evaluation metrics calculated on test dataset
        best model checkpoint, on which the metrics were calculated
    """

    logger.info("Loading the embedding model...")
    embeddings_model = cfg.model_class(cfg.pretrained_model_dir)

    clf = LogisticRegression(
        random_state=cfg.seed,
    )
    x_train, y_train = get_text_and_label(dataset=datasets.train_dataset)
    train_embeddings = embeddings_model(x_train)
    logger.info("Training logistic regression classifier...")
    clf.fit(train_embeddings, y_train)

    logger.info("Evaluating...")
    x_test, y_test = get_text_and_label(dataset=datasets.test_dataset)
    test_embeddings = embeddings_model(x_test)
    y_pred = clf.predict(test_embeddings)
    scores = {}
    scores["accuracy"] = accuracy_score(y_test, y_pred)
    scores["f1"] = f1_score(y_test, y_pred, average="macro")
    scores["f1_weighted"] = f1_score(y_test, y_pred, average="weighted")

    logger.info(scores)

    best_model_checkpoint = os.path.join(
        experiment_dir, "logistic_regression_model.pkl"
    )
    with open(best_model_checkpoint, mode="wb") as f:
        pickle.dump(clf, f)
    
    with open(os.path.join(experiment_dir, "test_results.json"), mode="w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4, sort_keys=True)

    return scores, best_model_checkpoint
    
    
def train_deep_head(
        datasets: Datasets, 
        experiment_dir: str,
        cfg: TrainCfg,
) -> tuple[dict[str, float], str]:
    """ 
    Train the deep classification head

    Args:
        datasets: train, val and test datasets to train on
        experiment_dir: the experiments directory to save the training outcomes
        cfg: the training configurations

    Returns:
        evaluation metrics calculated on test dataset
        best model checkpoint, on which the metrics were calculated
    """
    dataloaders = ClassificationTraining(
        embeddings_model=cfg.model_class,
        embeddings_model_checkpoint=cfg.pretrained_model_dir,
        cls_head=DeepHead,
        loss_fn=cfg.loss_fn,
        train_dataloader=datasets.train_dataloader,
        val_dataloader=datasets.val_dataloader,
        test_dataloader=datasets.test_dataloader,
        learning_rate=1e-3,
    )

    callbacks = [ModelCheckpoint(
        dirpath=experiment_dir,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )]
    l_logger = TensorBoardLogger(save_dir=experiment_dir, name="deep-cls-head")
    trainer = L.Trainer(
        max_epochs=cfg.num_epochs,
        callbacks=callbacks,
        accelerator="auto",
        logger=l_logger,
        log_every_n_steps=10,
    )
    trainer.fit(model=lightning_model)

    scores = trainer.test(ckpt_path="best")
    logger.info(scores)

    best_checkpoint_path = trainer._checkpoint_connector._select_ckpt_path(
        trainer.state.fn,
        "best",
        model_provided=False,
        model_connected=trainer.lightning_module is not None,
    )

    return scores, best_checkpoint_path


def train(cfg: TrainCfg ) -> tuple[dict[str, float], str]:
    """ 
    Train classification head for the text embedding model

    Args:
        cfg: the config parameters for model training

    Returns:
        evaluation metrics calculated on test set
        the experiment directory where the experiment was conducted
        the test dataset object
        best model checkpoint, on which metrics were computed
    """
    experiment_dir, run_id = set_experiment_dir(
        cfg.output_experiments_directory, 
        cfg=cfg)

    email_dataset = cfg.dataset_class(dataset_path=cfg.dataset_dir, limit_number_of_samples=20)

    train_d, val_d = split_dataset(ratio=cfg.train_ratio, dataset=email_dataset)
    val_d, test_d = split_dataset(ratio=cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio), dataset=val_d)

    datasets = Datasets(train_dataset=train_d, val_dataset=val_d, test_dataset=test_d)

    if cfg.classification_head == "logistic":
        scores, best_model_checkpoint = train_logistic_head(datasets=datasets, experiment_dir=experiment_dir, cfg=cfg)
    
    elif cfg.classification_head == "deep":
        scores, best_model_checkpoint = train_deep_head(datasets=datasets, experiment_dir=experiment_dir, cfg=cfg)

    else:
        raise NotImplementedError

    return scores, experiment_dir, datasets.test_dataset, best_model_checkpoint


def evaluate(cfg: TrainCfg, 
                experiment_dir: str,
                test_dataset: Dataset,
                best_model_checkpoint: str):
    
    """ 
    Evaluate the model.

    Args:
        cfg: the training parameters
        experiment_dir: the experiment directory
        test_dataset: the test dataset
        best_model_checkpoint: the best model checkpoint
    """
    model = ClassificationModel(
        embedding_cls=EmbeddingsModel,
        embedding_checkpoint=cfg.pretrained_model_dir,
        classification_checkpoint=DeepHead,
        classification_checkpoint=best_model_checkpoint,
    )

    test_dataset.shuffle()
    x_test, y_test = get_text_and_label(dataset=test_dataset)
    with torch.inference_mode():
        y_pred=model(x_test)

    scores = {}
    scores["accuracy"] = accuracy_score(y_test, y_pred)
    scores["f1"] = f1_score(y_test, y_pred, average="macro")
    scores["f1_weighted"] = f1_score(y_test, y_pred, average="weighted")
    scores["precision"] = precision_score(y_test, y_pred, average="macro")
    scores["recall"] = recall_score(y_test, y_pred, average="macro")
    scores["best_model_ckpt"] = best_model_checkpoint
    logger.info(f"Evaluation results: {scores}")

    with open(
        os.path.join(experiment_dir, "test_results.json"),
        mode="w",
        encoding="utf-8",
    ) as f:
        json.dump(scores, f, indent=4, sort_keys=True)

    
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred,
        cmap="Blues",
        ax=ax,
        display_labels=list(model.class_mappings.keys())
    )

    ax.set_title(label="Confusion Matrix")
    plt.savefig(os.path.join(experiment_dir, "confusion_matrix.png"), dpi=300)

    with torch.inference_mode():
        for idx, elem in enumerate(test_dataset):
            pred = model(elem.email_text)
            logger.info(f"pred={pred}, true={elem.email_cls}")
            if idx > 10:
                break

@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def run_pipe(cfg: DictConfig) -> None:
    train_cfg = TrainCfg(**dict(cfg.train))
    train(cfg=train_cfg)


if __name__ == "__main__":
    run_pipe()