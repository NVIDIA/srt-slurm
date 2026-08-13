# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cluster-level environment and ``${VAR}`` host passthrough."""

import pytest

from srtctl.core.config import resolve_config_with_defaults

SECRET = "hf_supersecret_do_not_leak"

BASE_CONFIG = {
    "name": "test-job",
    "model": {"path": "/models/test", "container": "test.sqsh", "precision": "fp8"},
    "resources": {"gpu_type": "h100", "agg_nodes": 1, "agg_workers": 1},
}


def _config(environment=None, **sections):
    config = {**BASE_CONFIG, **sections}
    if environment is not None:
        config["environment"] = environment
    return config


@pytest.fixture
def host_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    return SECRET


class TestClusterEnvironment:
    """srtslurm.yaml can carry environment variables for every job on the cluster."""

    def test_cluster_environment_applied(self):
        resolved = resolve_config_with_defaults(_config(), {"environment": {"HF_HOME": "/cluster/cache"}})
        assert resolved["environment"] == {"HF_HOME": "/cluster/cache"}

    def test_recipe_wins_per_key(self):
        """A recipe overrides one cluster variable without restating the others."""
        resolved = resolve_config_with_defaults(
            _config({"HF_HOME": "/recipe/cache"}),
            {"environment": {"HF_HOME": "/cluster/cache", "NCCL_DEBUG": "WARN"}},
        )
        assert resolved["environment"] == {"HF_HOME": "/recipe/cache", "NCCL_DEBUG": "WARN"}

    def test_no_cluster_config_leaves_recipe_alone(self):
        resolved = resolve_config_with_defaults(_config({"NCCL_DEBUG": "INFO"}), None)
        assert resolved["environment"] == {"NCCL_DEBUG": "INFO"}


class TestHostEnvPassthrough:
    """``VAR: ${VAR}`` forwards a host variable by name, never by value."""

    def test_reference_becomes_passthrough_name(self, host_token):
        resolved = resolve_config_with_defaults(_config({"HF_TOKEN": "${HF_TOKEN}"}), None)

        assert resolved["host_env_passthrough"] == ["HF_TOKEN"]
        # The entry is removed so nothing later exports the literal over the forwarded value.
        assert "HF_TOKEN" not in resolved["environment"]

    def test_secret_value_never_enters_the_config(self, host_token):
        """The whole point: the value must not reach anything that gets written to disk."""
        resolved = resolve_config_with_defaults(
            _config(
                {"HF_TOKEN": "${HF_TOKEN}"},
                backend={"type": "vllm", "decode_environment": {"HF_TOKEN": "${HF_TOKEN}"}},
            ),
            {"environment": {"HF_TOKEN": "${HF_TOKEN}"}},
        )
        assert SECRET not in repr(resolved)

    def test_reference_works_in_every_env_section(self, host_token, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        resolved = resolve_config_with_defaults(
            _config(
                {"HF_TOKEN": "${HF_TOKEN}"},
                backend={"type": "vllm", "decode_environment": {"OPENAI_API_KEY": "${OPENAI_API_KEY}"}},
                benchmark={"type": "manual", "env": {"HF_TOKEN": "${HF_TOKEN}"}},
            ),
            None,
        )
        assert resolved["host_env_passthrough"] == ["HF_TOKEN", "OPENAI_API_KEY"]
        assert resolved["backend"]["decode_environment"] == {}
        assert resolved["benchmark"]["env"] == {}

    def test_cluster_reference_is_forwarded_too(self, host_token):
        """The point of putting it in srtslurm.yaml: recipes stay free of secrets."""
        resolved = resolve_config_with_defaults(_config(), {"environment": {"HF_TOKEN": "${HF_TOKEN}"}})
        assert resolved["host_env_passthrough"] == ["HF_TOKEN"]

    def test_literal_values_are_untouched(self):
        resolved = resolve_config_with_defaults(_config({"NCCL_DEBUG": "INFO", "PROMPT": "cost is $5"}), None)
        assert resolved["environment"] == {"NCCL_DEBUG": "INFO", "PROMPT": "cost is $5"}
        assert resolved["host_env_passthrough"] == []

    def test_bare_dollar_is_not_a_reference(self):
        """Only ``${VAR}`` is special, so values containing a plain $ stay literal."""
        resolved = resolve_config_with_defaults(_config({"PASSWORD": "$ecret$tuff"}), None)
        assert resolved["environment"] == {"PASSWORD": "$ecret$tuff"}
        assert resolved["host_env_passthrough"] == []

    def test_unset_variable_fails_at_submit(self, monkeypatch):
        """Better a clear error now than an unauthorized response an hour into the job."""
        monkeypatch.delenv("SRTCTL_ABSENT_VAR", raising=False)
        with pytest.raises(ValueError, match="not set in the submitting shell"):
            resolve_config_with_defaults(_config({"SRTCTL_ABSENT_VAR": "${SRTCTL_ABSENT_VAR}"}), None)

    def test_renaming_is_rejected(self, host_token):
        """Forwarding by name cannot rename, so fail instead of silently doing nothing."""
        with pytest.raises(ValueError, match="under a different name"):
            resolve_config_with_defaults(_config({"MY_TOKEN": "${HF_TOKEN}"}), None)

    def test_embedded_reference_is_rejected(self, host_token):
        with pytest.raises(ValueError, match="must be the entire value"):
            resolve_config_with_defaults(_config({"HF_TOKEN": "Bearer ${HF_TOKEN}"}), None)


class TestSbatchExport:
    """Submission drops the host environment unless a variable was asked for."""

    @staticmethod
    def _load(environment=None):
        from srtctl.core.schema import SrtConfig

        return SrtConfig.Schema().load(resolve_config_with_defaults(_config(environment), None))

    def test_clean_environment_by_default(self):
        from srtctl.cli.submit import sbatch_export_spec

        config = self._load()
        assert config.host_env_passthrough == ()
        assert sbatch_export_spec(config) == "NONE"

    def test_requested_variables_are_named(self, host_token, monkeypatch):
        from srtctl.cli.submit import sbatch_export_spec

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        config = self._load({"HF_TOKEN": "${HF_TOKEN}", "OPENAI_API_KEY": "${OPENAI_API_KEY}"})

        assert config.host_env_passthrough == ("HF_TOKEN", "OPENAI_API_KEY")
        # Names only: SLURM reads the values from the submitting shell itself.
        assert sbatch_export_spec(config) == "HF_TOKEN,OPENAI_API_KEY"

    def test_secret_never_reaches_the_command_line(self, host_token):
        from srtctl.cli.submit import sbatch_export_spec

        assert SECRET not in sbatch_export_spec(self._load({"HF_TOKEN": "${HF_TOKEN}"}))
