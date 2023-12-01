import functools
import typing as typ

import omegaconf as omg
import torch
import transformers

from vod_configs.support import TweaksConfig
from vod_types.batch import RealmBatch, RealmOutput
from transformers import modeling_outputs
from vod_models.gradients import Gradients
from vod_models.encoder import VodEncoder
from vod_models.support import apply_tweaks

from .ranker import Ranker

AutoregressiveLanguageModel = (
    transformers.BlenderbotForConditionalGeneration | transformers.LlamaForCausalLM | transformers.OPTForCausalLM
)


class Realm(Ranker):
    """A Retrieval-augmented Language Model."""

    def __init__(
        self,
        encoder: VodEncoder,
        lm: AutoregressiveLanguageModel,
        gradients: Gradients,
        optimizer: None | dict | omg.DictConfig | functools.partial = None,
        scheduler: None | dict | omg.DictConfig | functools.partial = None,
        tweaks: None | dict | omg.DictConfig | TweaksConfig = None,
    ):
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            encoder=encoder,
            gradients=gradients,
            tweaks=tweaks,
        )

        # Prepare the language model with optional optimizations
        self.lm = apply_tweaks(lm, tweaks)  # type: ignore

    def evaluate(
        self,
        batch: RealmBatch,
        **kws: typ.Any,
    ) -> RealmOutput:  # noqa: ARG002
        """Run a forward pass and compute the gradients."""
        lm_logits = self._forward_lm(
            input_ids=batch.lm__input_ids,  # type: ignore
            attention_mask=batch.lm__attention_mask,  # type: ignore
        )
        encoded = self.encode(batch)
        return self.gradients(batch=batch, **encoded, lm_logits=lm_logits)

    def _forward_lm(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the language model."""
        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])
        attention_mask = attention_mask.view(-1, input_shape[-1])
        output: modeling_outputs.CausalLMOutputWithPast = self.lm(input_ids, attention_mask)
        lm_logits = output.logits
        return lm_logits.view(*input_shape, -1)
