# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`srtctl render`: the sbatch script `apply` would submit, written out for someone else to submit."""

import logging
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from srtctl.cli import submit
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig

RECIPE = {
    "name": "render-me",
    "model": {"path": "/models/test-model", "container": "/containers/test.sqsh", "precision": "fp8"},
    "resources": {
        "gpu_type": "h100",
        "gpus_per_node": 8,
        "prefill_nodes": 1,
        "decode_nodes": 1,
        "prefill_workers": 1,
        "decode_workers": 1,
    },
    "benchmark": {"type": "manual"},
}


@pytest.fixture
def recipe(tmp_path: Path) -> Path:
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(RECIPE))
    return path


def test_render_writes_a_self_contained_script(
    recipe: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    render_dir = tmp_path / "rendered"
    with patch.object(submit, "validate_setup"):
        returned = submit.submit_single(config_path=recipe, render_dir=render_dir, enforce_preflight=False)

    script = render_dir / "sbatch_script.sh"
    assert returned == str(script.resolve())
    assert capsys.readouterr().out.rstrip().splitlines()[-1] == str(script.resolve())
    assert script.stat().st_mode & stat.S_IXUSR

    body = script.read_text()
    assert "#SBATCH --nodes=2" in body
    # The job stages the recipe itself, right after it creates OUTPUT_DIR: apply's
    # post-sbatch copy never runs for a rendered script.
    mkdir_at = body.index('mkdir -p "${LOG_DIR}"')
    copy_at = body.index(f'cp "{render_dir.resolve()}"/*.yaml "${{OUTPUT_DIR}}/"')
    assert mkdir_at < copy_at < body.index("srtctl.cli.do_sweep")
    assert yaml.safe_load((render_dir / "config.yaml").read_text())["name"] == "render-me"


def test_render_stages_the_resolved_override_variant(recipe: Path, tmp_path: Path) -> None:
    render_dir = tmp_path / "rendered"
    resolved = dict(RECIPE, name="render-me-variant")
    with patch.object(submit, "validate_setup"):
        submit.submit_single(
            config_path=recipe,
            config=SrtConfig.from_yaml(recipe),
            render_dir=render_dir,
            variant_suffix="variant",
            source_config_path=recipe,
            runtime_config_text=yaml.safe_dump(resolved),
            enforce_preflight=False,
        )

    assert (render_dir / "config.yaml").exists()
    assert yaml.safe_load((render_dir / "config_variant.yaml").read_text())["name"] == "render-me-variant"
    assert 'do_sweep "${OUTPUT_DIR}/config_variant.yaml"' in (render_dir / "sbatch_script.sh").read_text()


def test_render_writes_the_placement_sidecar(recipe: Path, tmp_path: Path) -> None:
    import json

    render_dir = tmp_path / "rendered"
    with patch.object(submit, "validate_setup"):
        submit.submit_single(config_path=recipe, render_dir=render_dir, enforce_preflight=False)
    placement = json.loads((render_dir / "render.json").read_text())
    assert placement["total_nodes"] == 2
    assert placement["frontend_node_index"] == 0
    assert placement["client_node_index"] == 0
    assert placement["frontend_port"] == 8000
    assert placement["served_model_name"] == "test-model"
    assert placement["benchmark_type"] == "manual"
    assert placement["script"] == str((render_dir / "sbatch_script.sh").resolve())


@pytest.mark.parametrize("frontend", [False, True])
@pytest.mark.parametrize("client", [False, True])
@pytest.mark.parametrize("infra", [False, True])
@pytest.mark.parametrize("colocate", [False, True])
def test_planned_role_indices_match_from_slurm(frontend: bool, client: bool, infra: bool, colocate: bool) -> None:
    nodelist = [f"n{i}" for i in range(6)]
    with (
        patch("srtctl.core.runtime.get_slurm_nodelist", return_value=nodelist),
        patch("srtctl.core.runtime.get_slurm_het_nodelists", return_value=None),
    ):
        nodes = Nodes.from_slurm(
            frontend_dedicated_node=frontend,
            client_dedicated_node=client,
            etcd_nats_dedicated_node=infra,
            colocate_dedicated_nodes=colocate,
        )
    head, bench = Nodes.planned_role_indices(
        len(nodelist),
        frontend_dedicated_node=frontend,
        client_dedicated_node=client,
        etcd_nats_dedicated_node=infra,
        colocate_dedicated_nodes=colocate,
    )
    assert nodelist[head] == nodes.head
    assert nodelist[bench] == nodes.bench


def test_apply_script_does_not_stage_a_recipe(recipe: Path) -> None:
    body = submit.generate_minimal_sbatch_script(config=SrtConfig.from_yaml(recipe), config_path=recipe)
    assert "srtctl render" not in body
    assert '*.yaml "${OUTPUT_DIR}/"' not in body


def test_render_cli_rejects_sweeps_and_directories(recipe: Path, tmp_path: Path) -> None:
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(yaml.safe_dump(dict(RECIPE, sweep={"parameters": {}})))
    with (
        patch.object(submit, "validate_setup"),
        patch.object(submit.sys, "argv", ["srtctl", "render", "-f", str(sweep), "--to", str(tmp_path / "out")]),
        pytest.raises(SystemExit) as exit_info,
    ):
        submit.main()
    assert exit_info.value.code == 1

    with (
        patch.object(submit, "validate_setup"),
        patch.object(submit.sys, "argv", ["srtctl", "render", "-f", str(tmp_path), "--to", str(tmp_path / "out")]),
        pytest.raises(SystemExit) as exit_info,
    ):
        submit.main()
    assert exit_info.value.code == 1


def test_render_cli_prints_only_the_script_path_on_stdout(
    recipe: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    render_dir = tmp_path / "out"
    argv = ["srtctl", "render", "-f", str(recipe), "--to", str(render_dir), "--no-preflight"]
    with patch.object(submit, "validate_setup"), patch.object(submit.sys, "argv", argv):
        submit.main()
    out = capsys.readouterr().out.strip().splitlines()
    assert out == [str((render_dir / "sbatch_script.sh").resolve())]


def test_telemetry_accepts_a_manual_benchmark(recipe: Path, tmp_path: Path) -> None:
    data = dict(RECIPE)
    data["telemetry"] = {"enabled": True, "cpu_power": {"enabled": True, "source": "dcgm"}}
    path = tmp_path / "manual-telemetry.yaml"
    path.write_text(yaml.safe_dump(data))
    config = SrtConfig.from_yaml(path)
    assert config.benchmark.type == "manual"
    assert config.telemetry.cpu_power.enabled


def test_colliding_extra_mounts_are_reported(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real)
    model = tmp_path / "model"
    model.mkdir()
    image = tmp_path / "image.sqsh"
    image.write_text("")
    data = dict(RECIPE, model={"path": str(model), "container": str(image), "precision": "fp8"})
    data["extra_mount"] = [f"{alias}:/alias", f"{real}:/real"]
    path = tmp_path / "mounts.yaml"
    path.write_text(yaml.safe_dump(data))

    with (
        caplog.at_level(logging.WARNING, logger="srtctl.core.runtime"),
        patch("srtctl.core.runtime.get_slurm_nodelist", return_value=["n0", "n1"]),
        patch("srtctl.core.runtime.get_slurm_het_nodelists", return_value=None),
        patch("srtctl.core.runtime.get_hostname_ip", side_effect=lambda node, *_: node),
        patch.dict("os.environ", {"SLURM_JOB_ID": "1", "SLURM_NODELIST": "n[0-1]"}, clear=False),
    ):
        context = RuntimeContext.from_config(SrtConfig.from_yaml(path), job_id="1", log_dir_base=tmp_path / "logs")

    assert context.container_mounts[real.resolve()] == Path("/real")
    assert any("resolves to" in record.getMessage() for record in caplog.records)


def test_render_refuses_an_override_file_without_a_selector(recipe: Path, tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"
    override.write_text(yaml.safe_dump({"base": RECIPE, "override_a": {"name": "a"}, "override_b": {"name": "b"}}))
    with patch.object(submit, "validate_setup"), pytest.raises(ValueError, match="exactly one variant"):
        submit.submit_override(override, render_dir=tmp_path / "out", enforce_preflight=False)


def test_render_into_the_recipe_directory_does_not_copy_onto_itself(tmp_path: Path) -> None:
    recipe = tmp_path / "config.yaml"
    recipe.write_text(yaml.safe_dump(RECIPE))
    with patch.object(submit, "validate_setup"):
        submit.submit_single(config_path=recipe, render_dir=tmp_path, enforce_preflight=False)
    assert (tmp_path / "sbatch_script.sh").exists()
    assert yaml.safe_load(recipe.read_text())["name"] == "render-me"


def test_the_ready_marker_records_the_health_gate(tmp_path: Path) -> None:
    import json

    from srtctl.cli.mixins.benchmark_stage import SERVER_READY_FILENAME, write_server_ready_marker

    marker = write_server_ready_marker(tmp_path)
    assert marker == tmp_path / SERVER_READY_FILENAME
    assert json.loads(marker.read_text())["ready_at_unix"] > 0
    assert write_server_ready_marker(tmp_path / "missing" / "dir") is None
