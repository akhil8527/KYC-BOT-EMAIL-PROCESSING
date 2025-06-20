import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from transformers import AutoModel, AutoTokenizer, Trainer

from configurations.params import DATASET_PATH, EMBEDDING_MODEL_CHECKPOINT
from source.dataset import EmailDataset, TripletDataset, split_dataset

email_dataset = EmailDataset(dataset_path=DATASET_PATH, limit_number_of_samples=10)
train_d, val_d = split_dataset(ratio=0.3, dataset=email_dataset)
val_d, test_d = split_dataset(ratio=0.5, dataset=val_d)

train_d = TripletDataset(train_d)
val_d   = TripletDataset(val_d)
test_d  = TripletDataset(test_d)

def dataset_gen(dataset: TripletDataset):
    for anchor, positive, negative in dataset:
        yield {"anchor": anchor, "positive": positive, "negative": negative}


train_dataset = Dataset.from_generator(dataset_gen, gen_kwargs={"dataset": train_d})
val_dataset   = Dataset.from_generator(dataset_gen, gen_kwargs={"dataset": val_d})
test_dataset  = Dataset.from_generator(dataset_gen, gen_kwargs={"dataset": test_d})

model = SentenceTransformer(EMBEDDING_MODEL_CHECKPOINT)
loss  = MultipleNegativesRankingLoss(model=model)
config= LoraConfig(
    target_modules=["encoder.layer.23.output.dense"],
)
peft_model = get_peft_model(model, config)

optimizer = create_loraplus_optimizer(
    model=peft_model,
    optimizer_cls=torch.optim.Adam,
    lr=5e-5,
    loraplus_lr_ratio=16,
)
scheduler = None

trainer = SentenceTransformerTrainer(
    model=peft_model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss=loss,
    optimizers=(optimizer, scheduler),
)

trainer.train()
peft_model.save_pretrained("checkpoints\\peft")

test_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"],
    positives=test_dataset["positive"],
    negatives=test_dataset["negative"],
    similarity_fn_names=["cosine", "dot", "euclidean", "manhattan"],
)

current_res = test_evaluator(peft_model)
print(f"peft: {current_res}")

old_model = SentenceTransformer(EMBEDDING_MODEL_CHECKPOINT)
test_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"],
    positives=test_dataset["positive"],
    negatives=test_dataset["negative"],
    similarity_fn_names=["cosine", "dot", "euclidean", "manhattan"],
)
old_res = test_evaluator(old_model)
print(f"old: {old_res}")

""" 
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.merge_and_unload()
"""