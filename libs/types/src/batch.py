import typing as typ

import torch
from torch.utils import _pytree
from typing_extensions import Self, Type

from .mapping import MappingMixin

UNSET = object()

EXTRA_MODE = typ.Literal["raise", "ignore", "keep"]


class Batch(MappingMixin):
    """Adictionary like object behaving like a dictionary, but also accepts class attributes."""

    _extras: EXTRA_MODE = "raise"  # Specify how to handle extra attributes

    def __init__(self, *args: typ.Mapping[str, typ.Any], **kws: typ.Any):  # noqa: C901
        if len(args) > 0 and len(kws) > 0:
            raise ValueError
        if len(args) > 1:
            raise ValueError
        if len(args) == 1:
            kws = args[0]  # type: ignore

        # get the class attributes that don't support defaults (required attributes)
        required_attributes = set()
        for k in self.__class__.__annotations__:
            v = getattr(self.__class__, k, UNSET)
            if UNSET is v:
                required_attributes.add(k)

        # register attributes
        unknown_attributes = set()
        set_attributes = set()
        for k, v in kws.items():
            if k not in self.__class__.__annotations__:
                unknown_attributes.add(k)
                if self._extras != "keep":
                    continue
            setattr(self, k, v)
            set_attributes.add(k)
        if self._extras == "raise" and len(unknown_attributes):
            raise ValueError(f"Unknown attributes `{unknown_attributes}`.")

        # check if all required attributes are set
        missing_attributes = required_attributes - set_attributes
        if len(missing_attributes) > 0:
            raise ValueError(f"Missing required attributes `{missing_attributes}`.")

    @classmethod
    def cast(cls: Type[Self], data: typ.Mapping[str, typ.Any]) -> Self:
        """Cast input data as a `cls`."""
        if not isinstance(data, cls):
            return cls(data)

        return data

    def __repr__(self) -> str:
        attributes = ",\n".join(f"    {k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}(\n{attributes}\n)"

    def to_dict(self) -> dict[str, typ.Any]:
        """Return a dictionary representation of the object."""
        return self.__dict__.copy()


class RealmBatch(Batch):
    """Represents a tokenized batch for retrieval-augmented tasks."""

    _extras: EXTRA_MODE = "keep"  # Allow extra attributes

    # Language Model tokenized text
    lm__input_ids: None | torch.Tensor = None
    lm__attention_mask: None | torch.Tensor = None
    lm__token_type_ids: None | torch.Tensor = None  # Takes value 0/1/2 whether the token is context/question/answer
    # Query tokenized text
    query__input_ids: torch.Tensor
    query__attention_mask: torch.Tensor
    # Query extras
    query__id: str
    query__subset_ids: list[str]
    query__section_ids: list[str]
    query__language: None | str = None
    # Section tokenized text
    section__input_ids: torch.Tensor
    section__attention_mask: torch.Tensor
    # Section extras
    section__id: str
    section__subset_id: None | str = None
    section__language: None | str = None
    # Retrieval label & scores
    section__relevance: torch.Tensor
    section__idx: torch.Tensor
    section__score: torch.Tensor
    section__sparse: torch.Tensor
    section__dense: None | torch.Tensor = None
    # Priority sampling
    section__log_weight: torch.Tensor
    section__lse_pos: torch.Tensor
    section__lse_neg: torch.Tensor
    # Diagnostics
    diagnostics: dict[str, typ.Any] = {}


class RealmOutput(Batch):
    """Represents the output of a model."""

    loss: torch.Tensor
    # Score for each section in the batch
    retriever_scores: torch.Tensor
    # Diagnostic information
    diagnostics: dict[str, typ.Any] = {}


# NOTE: Register custom data structures in pytree nodes
#       See: https://github.com/pytorch/pytorch/issues/106690#issuecomment-1667272494
#       If not registered, you may get unexpected results when working with `torch.distributed` !
_pytree._register_pytree_node(
    RealmOutput,
    _pytree._dict_flatten,
    lambda values, context: RealmOutput(_pytree._dict_unflatten(values, context)),
)
_pytree._register_pytree_node(
    RealmBatch,
    _pytree._dict_flatten,
    lambda values, context: RealmBatch(_pytree._dict_unflatten(values, context)),
)
