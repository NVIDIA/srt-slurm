# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the v1 -> v2 layout migrator.

The loader no longer reads the v1 layout, so the migrator's contract is: a v1
document is rejected by the loader, and the migrated document loads. Every
fixture here asserts both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from srtctl.cli import submit as submit_cli
from srtctl.core.config import generate_override_configs, legacy_keys_present, resolve_config_with_defaults
from srtctl.core.migrate import migrate_recipe_text
from srtctl.core.schema import SrtConfig

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _loads(text: str) -> list[SrtConfig]:
    """Resolve and load every concrete recipe a document holds, exactly as the loader does."""
    raw = yaml.safe_load(text)
    variants = generate_override_configs(raw) if "base" in raw else [("", raw)]
    schema = SrtConfig.Schema()
    return [schema.load(resolve_config_with_defaults(cfg, None)) for _, cfg in variants]


def _migrated_loads_and_v1_does_not(v1_text: str) -> list[SrtConfig]:
    with pytest.raises(ValueError, match="srtctl migrate"):
        _loads(v1_text)
    migrated = migrate_recipe_text(v1_text).text
    raw = yaml.safe_load(migrated)
    for _, variant in generate_override_configs(raw) if "base" in raw else [("", raw)]:
        assert legacy_keys_present(variant) == [], legacy_keys_present(variant)
    return _loads(migrated)


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

    (config,) = _migrated_loads_and_v1_does_not(LEGACY)
    assert config.resources.num_prefill == 2
    assert config.frontend.orchestrator_placement == "first_decode"
    assert config.infra.etcd_nats_dedicated_node is True
    assert config.dynamo.hash == "abc1234"
    assert config.benchmark.client_placement == "last_decode"


def test_migrate_spells_shared_node_decode_as_colocate() -> None:
    # 1 node x 8 GPUs: 1 prefill x 4 + 1 decode x 4 fits on the shared node
    legacy = LEGACY.replace("  decode_nodes: 1", "  decode_nodes: 0  # share the prefill node").replace(
        "  prefill_workers: 2", "  prefill_workers: 1\n  gpus_per_prefill: 4"
    )
    result = migrate_recipe_text(legacy)
    doc = yaml.safe_load(result.text)
    assert doc["roles"]["decode"]["nodes"] == "colocate"
    # the derived split is written out: prefill kept its explicit 4, decode inherited it
    assert doc["roles"]["prefill"]["gpus"] == 4
    assert doc["roles"]["decode"]["gpus"] == 4
    assert list(doc["roles"]["decode"])[:3] == ["nodes", "workers", "gpus"]
    assert "decode_nodes" not in doc.get("resources", {})
    assert any("nodes: colocate" in note for note in result.notes)
    (config,) = _migrated_loads_and_v1_does_not(legacy)
    assert config.resources.decode_nodes == 0
    assert config.resources.gpus_per_prefill == 4
    assert config.resources.gpus_per_decode == 4


def test_migrate_is_idempotent_and_layout_folds_apply_to_schema_2_documents() -> None:
    once = migrate_recipe_text(LEGACY)
    twice = migrate_recipe_text(once.text)
    assert not twice.changed
    assert twice.notes == ()

    # Declaring schema: 2 on a v1 document does not make it load; migrate folds it anyway.
    legacy_v2 = "schema: 2\n" + LEGACY.split("\n", 1)[1]
    with pytest.raises(ValueError, match=r"pre-2\.0 \(v1\) layout"):
        _loads(legacy_v2)
    folded = migrate_recipe_text(legacy_v2)
    assert "roles" in yaml.safe_load(folded.text)
    assert len(_loads(folded.text)) == 1


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
    # And the variants still combine: a partially migrated file would collide on roles vs internal fields.
    configs = _migrated_loads_and_v1_does_not(text)
    assert len(configs) == 3
    assert configs[0].resources.num_agg == 1 and configs[0].resources.gpus_per_agg == 2
    assert [c.backend.sglang_config.aggregated["context-length"] for c in configs[1:]] == [2048, 8192]


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
    # This is the one case the migrator cannot finish: the leftover infra block is still rejected at load.
    assert legacy_keys_present(doc) == ["infra"]

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
    assert doc["base"]["services"] == [{"name": "etcd", "type": "etcd", "placement": {"node": "dedicated"}}]
    assert doc["override_shared"]["services"] == [
        {"name": "etcd", "type": "etcd", "placement": {"node": "infra"}},
    ]
    assert doc["override_deleted"]["services"] == doc["override_shared"]["services"]
    assert "infra" not in doc["override_deleted"]
    configs = _migrated_loads_and_v1_does_not(text)
    assert len(configs) == 2  # the two override variants; base alone is not a job
    assert all(c.infra.etcd_nats_dedicated_node is False for c in configs)


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
    (config,) = _migrated_loads_and_v1_does_not(MOONCAKE_LEGACY)
    assert config.backend.mooncake_kv_store is not None
    assert config.backend.mooncake_kv_store.container == "mooncake"
    assert config.backend.prefill_environment["MOONCAKE_GLOBAL_SEGMENT_SIZE"] == "4gb"


def test_dynamo_version_and_wheel_and_top_of_tree() -> None:
    head = "name: d\nmodel:\n  path: /m\n  container: /c\n  precision: bf16\n"
    assert yaml.safe_load(migrate_recipe_text(head + "dynamo:\n  version: '1.4.2'\n").text)["dynamo"] == {
        "source": {"pypi": "1.4.2"}
    }
    assert yaml.safe_load(migrate_recipe_text(head + "dynamo:\n  wheel: '1.5.0.dev1'\n").text)["dynamo"] == {
        "source": {"wheel": "1.5.0.dev1"}
    }
    result = migrate_recipe_text(head + "dynamo:\n  top_of_tree: true\n")
    doc = yaml.safe_load(result.text)
    assert doc["dynamo"] == {"top_of_tree": True}
    assert any("top_of_tree left as is" in note for note in result.notes)
    assert legacy_keys_present(doc) == []  # still a recipe key: the migrated document loads


def test_examples_are_already_current() -> None:
    """Every checked-in example is a schema 2 document with nothing left for the migrator to fold."""
    for path in sorted(EXAMPLES_DIR.rglob("*.yaml")):
        result = migrate_recipe_text(path.read_text())
        assert not result.changed, f"{path}: {result.notes}"


def test_cli_in_place_directory(tmp_path: Path, monkeypatch) -> None:
    for name in ("a.yaml", "b.yml"):
        (tmp_path / name).write_text(LEGACY)
    monkeypatch.setattr(sys, "argv", ["srtctl", "migrate", "--in-place", "-f", str(tmp_path)])
    submit_cli.main()
    for name in ("a.yaml", "b.yml"):
        doc = yaml.safe_load((tmp_path / name).read_text())
        assert doc["schema"] == 2 and "roles" in doc


def test_cli_in_place_directory_continues_past_an_unreadable_recipe(tmp_path: Path, monkeypatch, capsys) -> None:
    """One recipe with duplicate keys must not stop the rest of the directory from migrating."""
    good = tmp_path / "good.yaml"
    good.write_text(LEGACY)
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: dup\nmodel:\n  path: /m\n  container: /c\n  precision: bf16\nbackend:\n  type: sglang\n  type: vllm\n"
    )
    monkeypatch.setattr(sys, "argv", ["srtctl", "migrate", "-f", str(tmp_path), "--in-place"])
    with pytest.raises(SystemExit) as exc:
        submit_cli.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "duplicate key" in out
    assert "1 migrated, 1 not migrated" in out
    assert yaml.safe_load(good.read_text())["schema"] == 2
    assert "schema" not in bad.read_text()


def test_custom_benchmark_with_power_telemetry_keeps_its_concurrencies() -> None:
    """Power telemetry builds its measurement windows from benchmark.concurrencies for every
    type, so the per-type strip must leave it alone even for a custom client."""
    text = (
        "name: p\nmodel:\n  path: /m\n  container: /c\n  precision: fp4\n"
        "resources:\n  gpu_type: gb300\n  gpus_per_node: 4\n  agg_nodes: 1\n  agg_workers: 1\n  gpus_per_agg: 4\n"
        "backend:\n  type: trtllm\n"
        "benchmark:\n  type: custom\n  command: bash run.sh\n  concurrencies: '4'\n  use_chat_template: true\n"
        "telemetry:\n  enabled: true\n  dcgm_exporter:\n    container_image: dcgm\n    port: 9401\n"
    )
    result = migrate_recipe_text(text)
    doc = yaml.safe_load(result.text)
    assert doc["benchmark"] == {"type": "custom", "command": "bash run.sh", "concurrencies": "4"}
    assert "removed benchmark.use_chat_template (unused by type custom)" in result.notes
    (config,) = _migrated_loads_and_v1_does_not(text)
    assert config.benchmark.get_concurrency_list() == [4]


def test_migrate_folds_worker_criticality_into_roles() -> None:
    legacy = LEGACY.replace(
        "  decode_workers: 1", "  decode_workers: 1\n  decode_critical: false  # the probe kills decode workers"
    )
    result = migrate_recipe_text(legacy)
    doc = yaml.safe_load(result.text)
    assert doc["roles"]["decode"]["critical"] is False
    assert "critical" not in doc["roles"]["prefill"]
    assert "decode_critical" not in doc.get("resources", {})
    (config,) = _migrated_loads_and_v1_does_not(legacy)
    assert config.resources.worker_critical("decode") is False
    assert config.resources.worker_critical("prefill") is True
