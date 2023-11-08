from typing import Any, Optional, Union

import transformers
from typing_extensions import Literal, TypeAlias

AggMethod: TypeAlias = Literal["mean", "max", "cls", "none"]
NormMethod: TypeAlias = Literal["l1", "l2", "none"]
ActivationMethod: TypeAlias = Literal["tanh", "relu", "sigmoid", "relu"]
VodEncoderInputType: TypeAlias = Literal["query", "section"]


class VodPoolerConfig:
    """Configuration for a VOD head."""

    projection_size: Optional[int] = None
    output_activation: None | ActivationMethod = None
    output_norm: None | NormMethod = None
    agg_method: AggMethod = "mean"
    scaler: float = 1.0
    learn_scaler: bool = False

    def __init__(
        self,
        projection_size: Optional[int] = None,
        output_activation: None | ActivationMethod = None,
        output_norm: None | NormMethod = None,
        agg_method: AggMethod = "mean",
        scaler: float = 1.0,
        learn_scaler: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.projection_size = projection_size
        self.output_activation = output_activation
        self.output_norm = output_norm
        self.agg_method = agg_method
        self.scaler = scaler
        self.learn_scaler = learn_scaler

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict."""
        return {
            "projection_size": self.projection_size,
            "output_activation": self.output_activation,
            "output_norm": self.output_norm,
            "agg_method": self.agg_method,
            "scaler": self.scaler,
            "learn_scaler": self.learn_scaler,
        }


class VodEncoderConfigExtension:
    """Configuration for a VOD encoder."""

    model_type: str
    pooler: VodPoolerConfig

    def __init__(self, pooler: Optional[VodPoolerConfig] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if pooler is None:
            pooler = VodPoolerConfig()
        if isinstance(pooler, dict):
            pooler = VodPoolerConfig(**pooler)
        if not isinstance(pooler, VodPoolerConfig):
            raise TypeError(f"Expected pooler to be a `VodPoolerConfig`, got `{type(pooler)}`")
        self.pooler = pooler

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict."""
        if not isinstance(self, transformers.PretrainedConfig):
            raise TypeError(f"Expected self to be a transformers.PretrainedConfig, got {type(self)}")
        output = super().to_dict()  # type: ignore
        try:
            output["pooler"] = self.pooler.to_dict()
        except AttributeError:
            output["pooler"] = self.pooler
        return output


class VodBertEncoderConfig(VodEncoderConfigExtension, transformers.BertConfig):
    """Configuration for a VOD encoder."""

    model_type = "vod_bert_encoder"
    ...


class VodT5EncoderConfig(VodEncoderConfigExtension, transformers.T5Config):
    """Configuration for a VOD encoder."""

    model_type = "vod_t5_encoder"
    ...


class VodRobertaEncoderConfig(VodEncoderConfigExtension, transformers.RobertaConfig):
    """Configuration for a VOD encoder."""

    model_type = "vod_roberta_encoder"
    ...


class VodXLMRobertaEncoderConfig(VodEncoderConfigExtension, transformers.RobertaConfig):
    """Configuration for a VOD encoder."""

    model_type = "vod_xlm_roberta_encoder"
    ...


VodEncoderConfig = Union[
    VodBertEncoderConfig,
    VodT5EncoderConfig,
    VodRobertaEncoderConfig,
    VodXLMRobertaEncoderConfig,
]
