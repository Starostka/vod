import dataclasses
import typing as typ

import numpy as np
import vod_configs
import vod_datasets
from typing_extensions import Self, Type

from vod_configs.datasets import QueriesDatasetConfig, DatasetConfig, SectionsDatasetConfig
from vod_configs.search import HybridSearchFactoryConfig
from vod_datasets.interface import load_queries, load_sections
from vod_types.lazy_array import as_lazy_array
from vod_types.sequence import DictsSequence, Sequence
from vod_types.lazy_array import Array
from vod_search.base import ShardName

T = typ.TypeVar("T")
K = typ.TypeVar("K")


@dataclasses.dataclass(frozen=True)
class QueriesWithVectors:
    """Holds a dict of queries and their vectors."""

    queries: dict[str, tuple[ShardName, DictsSequence]]
    vectors: None | dict[str, Sequence[np.ndarray]]
    descriptor: None | str = None

    @classmethod
    def from_configs(
        cls: Type[Self],
        queries: list[QueriesDatasetConfig],
        vectors: None | dict[DatasetConfig, Array],
    ) -> Self:
        """Load a list of datasets from their respective configs."""
        descriptor = "+".join(sorted(cfg.identifier for cfg in queries))
        key_map = {cfg.fingerprint(): cfg for cfg in queries}
        queries_by_key = {key: (cfg.link, load_queries(cfg)) for key, cfg in key_map.items()}
        vectors_by_key = (
            {key: as_lazy_array(vectors[cfg]) for key, cfg in key_map.items()} if vectors is not None else None
        )
        return cls(
            descriptor=descriptor,
            queries=queries_by_key,  # type: ignore
            vectors=vectors_by_key,  # type: ignore
        )

    def __repr__(self) -> str:
        vec_dict = {k: _repr_vector_shape(v) for k, v in self.vectors.items()} if self.vectors else None
        return f"{type(self).__name__}(queries={self.queries}, vectors={vec_dict})"


@dataclasses.dataclass(frozen=True)
class SectionsWithVectors(typ.Generic[K]):
    """Holds a dict of sections and their vectors."""

    sections: dict[ShardName, DictsSequence]
    vectors: None | dict[ShardName, Sequence[np.ndarray]]
    search_configs: dict[ShardName, HybridSearchFactoryConfig]
    descriptor: None | str = None

    @classmethod
    def from_configs(
        cls: Type[Self],
        sections: list[SectionsDatasetConfig],
        vectors: None | dict[DatasetConfig, Array],
    ) -> Self:
        """Load a list of datasets from their respective configs."""
        descriptor = "+".join(sorted(cfg.identifier for cfg in sections))
        sections_by_shard_name = {cfg.identifier: load_sections(cfg) for cfg in sections}
        vectors_by_shard_name = (
            {cfg.identifier: as_lazy_array(vectors[cfg]) for cfg in sections} if vectors is not None else None
        )
        configs_by_shard_name = {cfg.identifier: cfg.search for cfg in sections}
        return cls(
            descriptor=descriptor,
            sections=sections_by_shard_name,  # type: ignore
            vectors=vectors_by_shard_name,  # type: ignore
            search_configs=configs_by_shard_name,  # type: ignore
        )

    def __repr__(self) -> str:
        vec_dict = {k: _repr_vector_shape(v) for k, v in self.vectors.items()} if self.vectors else None
        return f"{type(self).__name__}(sections={self.sections}, vectors={vec_dict})"


def _repr_vector_shape(x: None | Sequence[np.ndarray]) -> str:
    """Return a string representation of the vectors."""
    if x is None:
        return "None"
    dims = [len(x), *x[0].shape]
    return f"[{', '.join(map(str, dims))}]"
