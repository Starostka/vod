import torch
import torch.nn

from vod_types.batch import RealmBatch, RealmOutput

from .base import Gradients


class VodGradients(Gradients):
    """Variational Open-Domain Gradients for end-to-end trainng of Realms.

    NOTE: see https://arxiv.org/abs/2210.06345
    """

    def __call__(
        self,
        *,
        batch: RealmBatch,
        query_encoding: torch.Tensor,
        section_encoding: torch.Tensor,
        lm_logits: torch.Tensor,
    ) -> RealmOutput:
        """Computes the loss.

        TODO(vlievin): implement VOD gradients
        """
        raise NotImplementedError("Coming soon!")
