# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3 cluster defaults are merged before recipe schema validation."""

import copy

import pytest
from marshmallow import ValidationError

from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.schema import ReportingConfig

CLUSTER_S3 = {
    "bucket": "cluster-bucket",
    "prefix": "cluster-prefix",
    "region": "us-west-2",
    "endpoint_url": "https://storage.example.com",
    "exclude": ["*.tmp"],
    "archive": ["*.out"],
}


def _resolve(recipe: dict, cluster: dict | None) -> ReportingConfig | None:
    resolved = resolve_config_with_defaults({"schema": 2, **recipe}, cluster)
    reporting = resolved.get("reporting")
    return ReportingConfig.Schema().load(reporting) if reporting is not None else None


def test_prefix_only_recipe_inherits_bucket_endpoint_and_upload_policy() -> None:
    recipe = {"reporting": {"s3": {"prefix": "recipe-prefix"}}}
    cluster = {"reporting": {"s3": copy.deepcopy(CLUSTER_S3)}}
    original_recipe, original_cluster = copy.deepcopy(recipe), copy.deepcopy(cluster)

    reporting = _resolve(recipe, cluster)

    assert reporting is not None and reporting.s3 is not None
    assert reporting.s3.bucket == "cluster-bucket"
    assert reporting.s3.prefix == "recipe-prefix"
    assert reporting.s3.region == "us-west-2"
    assert reporting.s3.endpoint_url == "https://storage.example.com"
    assert reporting.s3.exclude == ["*.tmp"]
    assert reporting.s3.archive == ["*.out"]
    assert recipe == original_recipe and cluster == original_cluster


@pytest.mark.parametrize(
    "recipe", [{}, {"reporting": {}}, {"reporting": {"status": {"endpoint": "https://status.example.com"}}}]
)
def test_omitted_s3_inherits_cluster_settings(recipe: dict) -> None:
    reporting = _resolve(recipe, {"reporting": {"s3": CLUSTER_S3}})
    assert reporting is not None and reporting.s3 is not None
    assert reporting.s3.bucket == "cluster-bucket"
    assert reporting.s3.exclude == ["*.tmp"]


@pytest.mark.parametrize("policy", [[], ["recipe-pattern"]])
def test_explicit_lists_replace_cluster_policy(policy: list[str]) -> None:
    reporting = _resolve(
        {"reporting": {"s3": {"exclude": policy, "archive": policy}}}, {"reporting": {"s3": CLUSTER_S3}}
    )
    assert reporting is not None and reporting.s3 is not None
    assert reporting.s3.exclude == policy and reporting.s3.archive == policy


def test_null_fields_clear_inherited_values() -> None:
    reporting = _resolve(
        {"reporting": {"s3": {"prefix": None, "endpoint_url": None, "exclude": None, "archive": None}}},
        {"reporting": {"s3": CLUSTER_S3}},
    )
    assert reporting is not None and reporting.s3 is not None
    assert reporting.s3.prefix is None and reporting.s3.endpoint_url is None
    assert reporting.s3.exclude is None and reporting.s3.archive is None


@pytest.mark.parametrize("recipe", [{"reporting": None}, {"reporting": {"s3": None}}])
def test_null_reporting_or_s3_disables_inherited_upload(recipe: dict) -> None:
    reporting = _resolve(recipe, {"reporting": {"s3": CLUSTER_S3}})
    assert reporting is None or reporting.s3 is None


def test_s3_merge_does_not_inherit_other_reporting_blocks() -> None:
    reporting = _resolve(
        {"reporting": {"s3": {"prefix": "recipe-prefix"}}},
        {
            "reporting": {
                "s3": CLUSTER_S3,
                "status": {"endpoint": "https://status.example.com"},
                "ai_analysis": {"enabled": True},
            }
        },
    )
    assert reporting is not None and reporting.s3 is not None
    assert reporting.status is None and reporting.ai_analysis is None


@pytest.mark.parametrize("cluster", [None, {}, {"reporting": {}}])
def test_partial_recipe_without_cluster_bucket_fails_validation(cluster: dict | None) -> None:
    with pytest.raises(ValidationError, match="bucket"):
        _resolve({"reporting": {"s3": {"prefix": "recipe-prefix"}}}, cluster)


def test_standalone_recipe_still_works() -> None:
    reporting = _resolve({"reporting": {"s3": {"bucket": "recipe-bucket", "prefix": "recipe-prefix"}}}, None)
    assert reporting is not None and reporting.s3 is not None
    assert reporting.s3.bucket == "recipe-bucket" and reporting.s3.prefix == "recipe-prefix"


def test_recipe_can_override_cluster_bucket_and_endpoint() -> None:
    reporting = _resolve(
        {"reporting": {"s3": {"bucket": "recipe-bucket", "endpoint_url": "https://recipe.example.com"}}},
        {"reporting": {"s3": CLUSTER_S3}},
    )
    assert reporting is not None and reporting.s3 is not None
    assert reporting.s3.bucket == "recipe-bucket" and reporting.s3.endpoint_url == "https://recipe.example.com"
    assert reporting.s3.prefix == "cluster-prefix"
