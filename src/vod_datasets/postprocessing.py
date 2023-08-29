import functools
from typing import Any

import datasets
import numpy as np

from src import vod_configs
from src.vod_tools import pipes

DSET_DESCRIPTOR_KEY = "_DSET_ID_"
DSET_LINK_KEY = "_LINK_"
_UNDEFINED_LINK = object()


def postprocess_queries(
    dset: datasets.Dataset,
    identifier: str,
    config: vod_configs.DatasetOptions,
    link: None | str = None,
) -> datasets.Dataset:
    """Post-process `queries` dataset. E.g., format `text` using `template`."""
    dset = dset.map(
        pipes.Partial(
            pipes.template_pipe,
            template=config.templates.queries,
            input_keys=["query", "options", "language"],
            output_key="text",
            allow_missing=True,
        ),
        **_prep_map_kwargs(config.prep_map_kwargs, desc=f"{identifier}: Post-processing questions"),
    )

    return _postprocessing(dset, identifier=identifier, config=config, link=link)


def postprocess_sections(
    dset: datasets.Dataset,
    identifier: str,
    config: vod_configs.DatasetOptions,
) -> datasets.Dataset:
    """Post-process `queries` dataset."""
    dset = dset.map(
        pipes.Partial(
            pipes.template_pipe,
            template=config.templates.sections,
            input_keys=["title", "content", "language"],
            output_key="text",
        ),
        **_prep_map_kwargs(config.prep_map_kwargs, desc=f"{identifier}: Post-processing sections"),
    )
    return _postprocessing(dset, identifier=identifier, config=config)


def _postprocessing(
    dset: datasets.Dataset,
    identifier: str,
    config: vod_configs.DatasetOptions,
    link: None | str | object = _UNDEFINED_LINK,
) -> datasets.Dataset:
    dset = _take_subset(dset, config.subset_size)
    dset = _add_extras_attributes(dset, identifier=identifier, link=link)
    return dset


def _add_extras(
    row: dict[str, Any],  # noqa: ARG001
    *args: Any,
    extras: dict[str, Any],
    **kwds: Any,
) -> dict[str, Any]:
    return extras


def _add_extras_attributes(
    dataset: datasets.Dataset,
    identifier: str,
    link: None | str | object = _UNDEFINED_LINK,
) -> datasets.Dataset:
    """Add a descriptor to the dataset."""
    xtras = {DSET_DESCRIPTOR_KEY: identifier}
    if link is not _UNDEFINED_LINK:
        if link is not None and not isinstance(link, str):
            raise TypeError(f"Expected `link` to be a string or None, but got `{type(link)}`")
        xtras[DSET_LINK_KEY] = link  # type: ignore

    return dataset.map(
        functools.partial(_add_extras, extras=xtras),
        **_prep_map_kwargs({}, desc=f"{identifier}: adding extras attributess", batched=False),
    )


def _prep_map_kwargs(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    always_on = {"batched": True, "with_indices": True}
    return {**base, **always_on, **overrides}


def _take_subset(dset: datasets.Dataset, subset_size: None | int) -> datasets.Dataset:
    """Take a subset of the dataset."""
    if subset_size is None:
        return dset

    rgn = np.random.RandomState(0)

    # sample the subsets
    ids = rgn.choice(list(range(len(dset))), size=subset_size, replace=False)
    return dset.select(ids)


def combine_datasets(
    inputs: datasets.Dataset | list[datasets.Dataset | datasets.DatasetDict],
) -> datasets.Dataset:
    """Combine a list of datasets into a single dataset."""
    if isinstance(inputs, datasets.Dataset):
        return inputs

    if isinstance(inputs, list):
        inputs = [combine_datasets(d) for d in inputs]  # type: ignore
        return datasets.concatenate_datasets(inputs)  # type: ignore

    if isinstance(inputs, (datasets.DatasetDict, dict)):
        return combine_datasets(list(inputs.values()))

    raise TypeError(f"Unexpected type `{type(inputs)}`")
