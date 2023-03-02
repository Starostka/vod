from __future__ import annotations

from typing import Union, Iterable, Optional

import pytorch_lightning as pl
from fsspec import Callback
from omegaconf import DictConfig
from pytorch_lightning.loggers import Logger


class Trainer(pl.Trainer):
    """A Small wrapper around the Trainer class to allow
    for more flexible `logger` and `callbacks` arguments."""

    def __init__(
        self,
        *args,
        logger: Union[Logger, Iterable[Logger], bool] = True,
        callbacks: Optional[Union[dict[str, Callback], list[Callback], Callback]] = None,
        **kwargs,
    ):
        if isinstance(logger, (dict, DictConfig)):
            logger = list(logger.values())

        if isinstance(callbacks, (DictConfig, list)):
            callbacks = list(callbacks.values())

        super().__init__(*args, logger=logger, callbacks=callbacks, **kwargs)
