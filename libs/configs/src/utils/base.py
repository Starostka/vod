import typing as typ

import pydantic


class StrictModel(pydantic.BaseModel):
    """A pydantic model with strict configuration (immutable, doesn't accept extra args)."""

    class Config:
        """Pydantic configuration."""

        extra = "forbid"
        frozen = True
        from_attributes = True


class AllowMutations:
    """Temporarily allow mutations on a model."""

    def __init__(self, model: pydantic.BaseModel) -> None:
        self.model = model
        self._is_frozen = model.model_config.get("frozen", None)

    def __enter__(self) -> pydantic.BaseModel:
        self.model.model_config["frozen"] = False
        return self.model

    def __exit__(self, *args: typ.Any, **kws: typ.Any) -> None:
        if self._is_frozen is None:
            self.model.model_config.pop("frozen", None)
        else:
            self.model.model_config["frozen"] = self._is_frozen
