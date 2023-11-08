import pathlib

import lightning as L
import torch
from lightning.fabric import wrappers as fabric_wrappers

from .trainer_state import TrainerState

TRAINER_STATE_PATH_FNAME = "state-trainer.json"
MODEL_STATE_FNAME = "state-model.pt"
OPTIMIZER_STATE_FNAME = "state-otimizer.pt"
SCHEDULER_STATE_FNAME = "state-scheduler.pt"


def save_training_state(
    fabric: L.Fabric,
    checkpoint_path: str | pathlib.Path,
    optimizer: None | torch.optim.Optimizer = None,
    model: None | torch.nn.Module = None,
    scheduler: None | torch.optim.lr_scheduler.LRScheduler = None,
    trainer_state: None | TrainerState = None,
) -> None:
    """Save the training state."""
    checkpoint_path = pathlib.Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save the Trainer state
    if trainer_state is not None:
        with open(checkpoint_path / TRAINER_STATE_PATH_FNAME, "w") as f:
            f.write(trainer_state.model_dump_json(indent=2))

    # Save the model state
    if model is not None:
        fabric.save(checkpoint_path / MODEL_STATE_FNAME, model.state_dict())

    # Save the optimizer state
    if optimizer is not None:
        fabric.save(checkpoint_path / OPTIMIZER_STATE_FNAME, optimizer.state_dict())

    # Save the scheduler state
    if scheduler is not None:
        fabric.save(checkpoint_path / SCHEDULER_STATE_FNAME, scheduler.state_dict())


def load_training_state(
    fabric: L.Fabric,
    checkpoint_path: str | pathlib.Path,
    optimizer: None | torch.optim.Optimizer = None,
    module: None | torch.nn.Module = None,
    scheduler: None | torch.optim.lr_scheduler.LRScheduler = None,
    trainer_state: None | TrainerState = None,
) -> None:
    """Load the training state."""
    checkpoint_path = pathlib.Path(checkpoint_path)

    # Load the Trainer state
    if trainer_state is not None:
        with open(pathlib.Path(checkpoint_path) / TRAINER_STATE_PATH_FNAME, "r") as f:
            loaded_trainer_state = TrainerState.model_validate_json(f.read())
            trainer_state.__dict__ = loaded_trainer_state.__dict__

    # Load the model state
    if module is not None:
        module.load_state_dict(fabric.load(checkpoint_path / MODEL_STATE_FNAME))

    # Load the optimizer state
    if optimizer is not None:
        optimizer.load_state_dict(fabric.load(checkpoint_path / OPTIMIZER_STATE_FNAME))

    # Load the scheduler state
    if scheduler is not None:
        scheduler.load_state_dict(fabric.load(checkpoint_path / SCHEDULER_STATE_FNAME))


def _unwrap_model(model: None | fabric_wrappers._FabricModule | torch.nn.Module) -> None | torch.nn.Module:
    if fabric_wrappers.is_wrapped(model):
        # TODO: remove this once this is handled in Fabric
        model = fabric_wrappers._unwrap_objects(model)
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            model = model.module
    return model
