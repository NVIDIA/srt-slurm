# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the v1 -> v2 layout migrator and the golden-equality check."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from srtctl.cli import submit as submit_cli
from srtctl.core.migrate import migrate_recipe_text, verify_migration_text

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

LEGACY = """\
# A v1 recipe with everything the migrator folds.
name: legacy
model:
  path: /m
  container: /c.sqsh
  precision: bf16
resources:
  gpu_type: h100         # keep me
  gpus_per_node: 8
  prefill_nodes: 1       # one prefill node
  prefill_workers: 2
  decode_nodes: 1
  decode_workers: 1
  gpus_per_decode: 4
frontend:
  type: dynamo
  orchestrator_placement: first_decode
dynamo:
  install: true
  hash: "abc1234"        # pinned build
  cargo_patches:
    - 'x = 1'
backend:
  type: sglang
  prefill_environment:
    PYTHONUNBUFFERED: "1"
  decode_environment:
    SGLANG_X: "2"
  sglang_config:
    prefill:
      tensor-parallel-size: 4   # tp
    decode:
      tensor-parallel-size: 4
infra:
  etcd_nats_dedicated_node: true
benchmark:
  type: gsm8k
  num_examples: 100
  isl: 1024              # never read by gsm8k
  client_placement: last_decode
"""


def test_migrate_folds_roles_placement_source_and_strips_unused_benchmark_fields() -> None:
    result = migrate_recipe_text(LEGACY)
    doc = yaml.safe_load(result.text)

    assert doc["schema"] == 2
    assert doc["roles"] == {
        "prefill": {
            "nodes": 1,
            "workers": 2,
            "env": {"PYTHONUNBUFFERED": "1"},
            "args": {"tensor-parallel-size": 4},
        },
        "decode": {"nodes": 1, "workers": 1, "gpus": 4, "env": {"SGLANG_X": "2"}, "args": {"tensor-parallel-size": 4}},
    }
    assert doc["resources"] == {"gpu_type": "h100", "gpus_per_node": 8}
    assert "backend" not in doc
    assert doc["engine"] == "sglang"
    assert doc["frontend"] == {"type": "dynamo", "placement": {"node": "first_decode"}}
    assert "infra" not in doc
    assert doc["services"] == [
        {"name": "etcd", "type": "etcd", "placement": {"node": "dedicated"}},
        {"name": "nats", "type": "nats", "placement": {"node": "dedicated"}},
    ]
    assert doc["dynamo"] == {"install": True, "source": {"rev": "abc1234", "patches": ["x = 1"]}}
    assert doc["benchmark"] == {"type": "gsm8k", "num_examples": 100, "placement": {"node": "last_decode"}}
    assert "removed benchmark.isl (unused by type gsm8k)" in result.notes

    # Comments travel with their keys.
    assert "# keep me" in result.text
    assert "nodes: 1       # one prefill node" in result.text or "# one prefill node" in result.text
    assert "# pinned build" in result.text
    assert "# tp" in result.text
    # engine takes backend's place and roles follows it.
    keys = list(doc)
    assert keys.index("roles") == keys.index("engine") + 1


def test_migrate_is_idempotent_and_layout_folds_apply_to_schema_2_documents() -> None:
    once = migrate_recipe_text(LEGACY)
    twice = migrate_recipe_text(once.text)
    assert not twice.changed
    assert twice.notes == ()

    legacy_v2 = "schema: 2\n" + LEGACY.split("\n", 1)[1]
    folded = migrate_recipe_text(legacy_v2)
    assert "roles" in yaml.safe_load(folded.text)


def test_migrate_override_file_folds_every_variant() -> None:
    text = """\
base:
  name: o
  model:
    path: /m
    container: /c.sqsh
    precision: bf16
  resources:
    gpu_type: h100
    gpus_per_node: 8
    agg_nodes: 1
    agg_workers: 2
    gpus_per_agg: 1
  backend:
    type: sglang
    sglang_config:
      aggregated:
        tensor-parallel-size: 1
  benchmark:
    type: sa-bench
    isl: 128
    osl: 128
    concurrencies: "4"
override_tp2:
  resources:
    agg_workers: 1
    gpus_per_agg: 2
  backend:
    sglang_config:
      aggregated:
        tensor-parallel-size: 2
zip_override_ctx:
  backend:
    sglang_config:
      aggregated:
        context-length: [2048, 8192]
"""
    doc = yaml.safe_load(migrate_recipe_text(text).text)
    assert doc["base"]["roles"]["agg"] == {"nodes": 1, "workers": 2, "gpus": 1, "args": {"tensor-parallel-size": 1}}
    assert "backend" not in doc["override_tp2"]
    assert doc["base"]["engine"] == "sglang"
    assert doc["override_tp2"]["roles"]["agg"] == {"workers": 1, "gpus": 2, "args": {"tensor-parallel-size": 2}}
    assert doc["zip_override_ctx"]["roles"]["agg"] == {"args": {"context-length": [2048, 8192]}}
    # And the variants still combine: a partially migrated file would collide on roles vs legacy fields.
    verified = verify_migration_text(text)
    assert verified.status == "ok", verified.detail
    assert verified.variants == 3


def test_infra_false_is_dropped_and_payload_becomes_a_nats_option() -> None:
    head = "name: i\nmodel:\n  path: /m\n  container: /c\n  precision: bf16\n"
    doc = yaml.safe_load(migrate_recipe_text(head + "infra:\n  etcd_nats_dedicated_node: false\n").text)
    assert "infra" not in doc and "services" not in doc

    doc = yaml.safe_load(migrate_recipe_text(head + "infra:\n  nats_max_payload_mb: 24\n").text)
    assert doc["services"] == [{"name": "nats", "type": "nats", "options": {"max_payload_mb": 24}}]


def test_infra_under_a_static_frontend_is_not_turned_into_services() -> None:
    """Declaring etcd/nats would launch a discovery plane nothing uses; the v1 flag still reserves a node."""
    head = "name: i\nmodel:\n  path: /m\n  container: /c\n  precision: bf16\nfrontend:\n  type: sglang\n"
    result = migrate_recipe_text(head + "infra:\n  nats_max_payload_mb: 8\n  etcd_nats_dedicated_node: true\n")
    doc = yaml.safe_load(result.text)
    assert "services" not in doc
    assert doc["infra"] == {"etcd_nats_dedicated_node": True}
    assert any("left as is" in note for note in result.notes)

    doc = yaml.safe_load(migrate_recipe_text(head + "infra: { nats_max_payload_mb: 8 }\n").text)
    assert "infra" not in doc and "services" not in doc


def test_infra_in_override_variants_round_trips() -> None:
    """A base `true` undone by an override `false` or `null` has to stay undone after migration."""
    text = """\
base:
  name: o
  model:
    path: /m
    container: /c.sqsh
    precision: bf16
  resources:
    gpu_type: h100
    gpus_per_node: 8
    agg_nodes: 1
    agg_workers: 2
    gpus_per_agg: 1
  backend:
    type: sglang
  infra:
    etcd_nats_dedicated_node: true
  benchmark:
    type: sa-bench
    isl: 128
    osl: 128
    concurrencies: "4"
override_shared:
  infra:
    etcd_nats_dedicated_node: false
override_deleted:
  infra: null
"""
    doc = yaml.safe_load(migrate_recipe_text(text).text)
    assert doc["base"]["services"][0]["placement"] == {"node": "dedicated"}
    assert doc["override_shared"]["services"] == [
        {"name": "etcd", "type": "etcd", "placement": {"node": "infra"}},
        {"name": "nats", "type": "nats", "placement": {"node": "infra"}},
    ]
    assert doc["override_deleted"]["services"] == doc["override_shared"]["services"]
    assert "infra" not in doc["override_deleted"]
    verified = verify_migration_text(text)
    assert verified.status == "ok", verified.detail
    assert verified.variants == 2  # the two override variants; base alone is not a job


MOONCAKE_LEGACY = """\
name: mc
model:
  path: /m
  container: /c.sqsh
  precision: bf16
resources:
  gpu_type: h100
  gpus_per_node: 8
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 1
  decode_workers: 1
backend:
  type: sglang
  prefill_environment:
    MOONCAKE_GLOBAL_SEGMENT_SIZE: "0"    # the store owns the segments
  mooncake_kv_store:
    container: mooncake
    master_extra_args: [--nof_eviction_high_watermark_ratio=0.9]
    env:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_GLOBAL_SEGMENT_SIZE: 4gb
  sglang_config:
    prefill:
      disaggregation-transfer-backend: mooncake
    decode:
      disaggregation-transfer-backend: mooncake
benchmark:
  type: manual
"""


def test_mooncake_kv_store_becomes_a_master_service_and_role_env() -> None:
    result = migrate_recipe_text(MOONCAKE_LEGACY)
    doc = yaml.safe_load(result.text)
    assert "backend" not in doc
    assert doc["engine"] == "sglang"
    assert doc["services"] == [
        {
            "name": "mooncake-master",
            "type": "mooncake-master",
            "container": "mooncake",
            "args": ["--nof_eviction_high_watermark_ratio=0.9"],
        }
    ]
    # Mooncake env lands on every role; a Mooncake value beats the role's own, as at launch.
    assert doc["roles"]["prefill"]["env"] == {"MOONCAKE_GLOBAL_SEGMENT_SIZE": "4gb", "MOONCAKE_PROTOCOL": "rdma"}
    assert doc["roles"]["decode"]["env"] == {"MOONCAKE_PROTOCOL": "rdma", "MOONCAKE_GLOBAL_SEGMENT_SIZE": "4gb"}
    assert "# the store owns the segments" in result.text
    verified = verify_migration_text(MOONCAKE_LEGACY)
    assert verified.status == "ok", verified.detail


def test_dynamo_version_and_wheel_and_top_of_tree() -> None:
    head = "name: d\nmodel:\n  path: /m\n  container: /c\n  precision: bf16\n"
    assert yaml.safe_load(migrate_recipe_text(head + "dynamo:\n  version: '1.4.2'\n").text)["dynamo"] == {
        "source": {"pypi": "1.4.2"}
    }
    assert yaml.safe_load(migrate_recipe_text(head + "dynamo:\n  wheel: '1.5.0.dev1'\n").text)["dynamo"] == {
        "source": {"wheel": "1.5.0.dev1"}
    }
    result = migrate_recipe_text(head + "dynamo:\n  top_of_tree: true\n")
    assert yaml.safe_load(result.text)["dynamo"] == {"top_of_tree": True}
    assert any("top_of_tree left as is" in note for note in result.notes)


def test_verify_reports_identical_and_mismatched() -> None:
    ok = verify_migration_text(LEGACY)
    assert ok.status == "ok", ok.detail
    assert ok.variants == 1

    # A recipe the v1 loader itself rejects is skipped, not counted against the migrator.
    skipped = verify_migration_text(
        "name: x\nmodel:\n  path: /m\n  container: /c\n  precision: bf16\nbenchmark:\n  type: nope\n"
    )
    assert skipped.status == "skipped"
    assert "does not load" in skipped.detail


def test_every_example_is_golden() -> None:
    for path in sorted(EXAMPLES_DIR.rglob("*.yaml")):
        outcome = verify_migration_text(path.read_text(), path)
        assert outcome.status == "ok", f"{path}: {outcome.detail}"


def test_cli_verify_directory(tmp_path: Path, monkeypatch, capsys) -> None:
    (tmp_path / "a.yaml").write_text(LEGACY)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.yaml").write_text(LEGACY.replace("name: legacy", "name: other"))
    monkeypatch.setattr(sys, "argv", ["srtctl", "migrate", "--verify", "-f", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        submit_cli.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "2 identical, 0 mismatched" in out
    assert "a.yaml" in out and "b.yaml" in out


def test_cli_in_place_directory(tmp_path: Path, monkeypatch) -> None:
    for name in ("a.yaml", "b.yml"):
        (tmp_path / name).write_text(LEGACY)
    monkeypatch.setattr(sys, "argv", ["srtctl", "migrate", "--in-place", "-f", str(tmp_path)])
    submit_cli.main()
    for name in ("a.yaml", "b.yml"):
        doc = yaml.safe_load((tmp_path / name).read_text())
        assert doc["schema"] == 2 and "roles" in doc
