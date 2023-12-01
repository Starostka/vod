import typing

import datasets
from vod_configs.datasets import BaseDatasetConfig, DatasetConfig, DatasetLoader, QueriesDatasetConfig, SectionsDatasetConfig
from vod_datasets import rosetta

from .postprocessing import (
    combine_datasets,
    postprocess_queries,
    postprocess_sections,
)


def _load_one_dataset(
    name_or_path: str | DatasetLoader,
    subset: str | None = None,
    split: str | None = None,
    **kws: typing.Any,
) -> datasets.Dataset | datasets.DatasetDict:
    if isinstance(name_or_path, str):
        data = datasets.load_dataset(name_or_path, subset, split=split, **kws)
        if isinstance(data, (datasets.IterableDataset, datasets.IterableDatasetDict)):
            raise NotImplementedError(f"`{type(data)}` not supported.")

        return data

    # Else, assume a `configs.DatasetLoader` and try to call it
    try:
        return name_or_path(subset=subset, split=split, **kws)
    except Exception as e:
        raise RuntimeError(
            f"Failed to use `{name_or_path}` as a callable following the `{DatasetLoader}` protocol."
        ) from e


def _load_dataset_from_config(config: BaseDatasetConfig, **kws: typing.Any) -> datasets.Dataset:
    """Load the dataset, process it according to the prompt template and return a HF dataset."""
    subsets = config.subsets or [None]
    loaded_subsets = [_load_one_dataset(config.name_or_path, subset, split=config.split) for subset in subsets]
    return combine_datasets(loaded_subsets)


def load_queries(config: QueriesDatasetConfig) -> datasets.Dataset:
    """Load a queries dataset."""
    dset = _load_dataset_from_config(config)
    dset = rosetta.transform(dset, output="queries")
    dset = postprocess_queries(dset, config.identifier, config=config.options)
    return dset


def load_sections(config: SectionsDatasetConfig) -> datasets.Dataset:
    """Load a sections dataset."""
    dset = _load_dataset_from_config(config)
    dset = rosetta.transform(dset, output="sections")
    dset = postprocess_sections(dset, config.identifier, config=config.options)
    return dset


def load_dataset(config: DatasetConfig) -> datasets.Dataset:
    """Load a dataset."""
    if isinstance(
        config,
        QueriesDatasetConfig,
    ):
        return load_queries(config)
    if isinstance(
        config,
        SectionsDatasetConfig,
    ):
        return load_sections(config)
    raise TypeError(f"Unexpected config type `{type(config)}`")
