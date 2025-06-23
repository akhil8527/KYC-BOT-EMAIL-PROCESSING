import hydra
import logging

from omegaconf import DictConfig, OmegaConf

from source.train_classifier import train, evaluate, TrainCfg

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def run_pipe(cfg: DictConfig) -> None:
    """ 
    Run the text classification head training and then the evaluation functions

    Args:
        cfg: the training configurations
    """

    logger.info(OmegaConf.to_yaml(cfg=cfg))
    train_cfg = TrainCfg(**dict(cfg.train))
    scores, experiment_dir, test_dataset, best_model_checkpoint = train(cfg=train_cfg)
    evaluate(
        cfg=train_cfg,
        experiment_dir=experiment_dir,
        test_dataset=test_dataset,
        best_model_checkpoint=best_model_checkpoint,
    )


if __name__ == "__main__":
    run_pipe()