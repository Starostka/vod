import numbers
import os
import typing as typ
from pathlib import Path

import datasets
import faiss
import lightning as L
import loguru
import numba
import omegaconf
import torch
import transformers
import vod_models
import yaml
from lightning.pytorch.loggers.wandb import WandbLogger
from vod_tools.misc.config import config_to_flat_dict

T = typ.TypeVar("T")


def _identity(value: T) -> T:
    """Return the same value."""
    return value


def _cast_hps(value: typ.Any) -> str:  # noqa: ANN401
    """Cast a value to a string."""
    formatter = {
        numbers.Number: _identity,
        str: _identity,
        Path: lambda x: str(x.absolute()),
    }.get(type(value), str)
    return formatter(value)


def set_training_context() -> None:
    """Set the general context for torch, datasets, etc."""
    omp_num_threads = os.environ.get("OMP_NUM_THREADS", None)
    if omp_num_threads is not None:
        loguru.logger.warning(f"OMP_NUM_THREADS={omp_num_threads}")
        faiss.omp_set_num_threads(int(omp_num_threads))
        torch.set_num_threads(int(omp_num_threads))
        numba.set_num_threads(int(omp_num_threads))

    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()  # type: ignore
    torch.set_float32_matmul_precision("medium")


def get_model_stats(ranker: vod_models.Ranker) -> dict[str, typ.Any]:
    """Get some stats about the model."""
    try:
        output_shape = list(ranker.encoder.get_encoding_shape())
    except Exception as exc:
        output_shape = str(exc)
    return {
        "n_trainable_params": sum(p.numel() for p in ranker.parameters() if p.requires_grad),
        "n_total_params": sum(p.numel() for p in ranker.parameters()),
        "output_shape": output_shape,
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),  # type: ignore
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),  # type: ignore
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),  # type: ignore
    }


def log_config(
    config: dict[str, typ.Any] | omegaconf.DictConfig,
    exp_dir: Path,
    extras: dict[str, typ.Any],
    fabric: L.Fabric,
) -> None:
    """Log the config as hyperparameters and save it locally."""
    config_path = Path(exp_dir, "config.yaml")
    all_data: dict[str, typ.Any] = omegaconf.OmegaConf.to_container(config, resolve=True)  # type: ignore
    all_data.update(extras)
    with config_path.open("w") as f:
        yaml.dump(all_data, f)

    for logger in fabric.loggers:
        if isinstance(logger, WandbLogger):
            flat_config = config_to_flat_dict(all_data, sep="/")
            flat_config = {k: _cast_hps(v) for k, v in flat_config.items()}
            logger.experiment.config.update(flat_config)
