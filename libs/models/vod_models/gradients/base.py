import abc

import torch
import torch.nn

from vod_types.batch import RealmBatch, RealmOutput


class Gradients(abc.ABC):
    """Base class for the gradients layer. The gradients layer is a pure function (no torch params)."""

    @abc.abstractmethod
    def __call__(
        self,
        *,
        batch: RealmBatch,
        query_encoding: torch.Tensor,  # the encoding of the queries
        section_encoding: torch.Tensor,  # the encoding of the documents/sections
        lm_logits: None | torch.Tensor,  # the logits of the language model
    ) -> RealmOutput:
        """Compute the gradients/loss."""
        raise NotImplementedError
