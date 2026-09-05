import subprocess
from pathlib import Path

import yaml

ASSET_DIR = Path(__file__).parents[1] / "configs" / "accuracy" / "minimax-m3"


def _load(name: str) -> dict:
    return yaml.safe_load((ASSET_DIR / name).read_text())


def _assert_minimax_reasoning(config: dict) -> None:
    endpoint = config["target"]["api_endpoint"]
    adapter = endpoint["adapter_config"]
    assert adapter["use_reasoning"] is True
    assert adapter["params_to_add"] == {
        "chat_template_kwargs": {"thinking_mode": "enabled"}
    }
    reasoning = next(
        item["config"]
        for item in adapter["interceptors"]
        if item["name"] == "reasoning"
    )
    assert reasoning["start_reasoning_token"] == "<mm:think>"
    assert reasoning["end_reasoning_token"] == "</mm:think>"
    assert endpoint["url"] == "__SRT_TARGET_URL__"


def test_mmlu_pro_aa_v3_sampling_and_methodology() -> None:
    config = _load("mmlu_pro_aa_v3.eval-factory.yaml")
    params = config["config"]["params"]
    assert config["config"]["type"] == "mmlu_pro_aa_v3"
    assert params["temperature"] == 1.0
    assert params["top_p"] == 0.95
    assert params["max_new_tokens"] == 65536
    assert params["extra"]["n_samples"] == 1
    _assert_minimax_reasoning(config)


def test_aa_lcr_sampling_repeats_and_authorized_judge() -> None:
    config = _load("ns_aa_lcr.eval-factory.yaml")
    params = config["config"]["params"]
    judge = params["extra"]["judge"]
    assert config["config"]["type"] == "ns_aa_lcr"
    assert params["temperature"] == 1.0
    assert params["top_p"] == 0.95
    assert params["max_new_tokens"] == 65536
    assert params["extra"]["num_repeats"] == 16
    assert judge["model_id"] == "nvidia/qwen/eccn-qwen-235b"
    assert judge["api_key"] == "INFERENCE_API_KEY"
    assert "nvapi-" not in (ASSET_DIR / "ns_aa_lcr.eval-factory.yaml").read_text()
    _assert_minimax_reasoning(config)


def test_nel_runner_is_valid_shell_and_does_not_enable_xtrace() -> None:
    script = ASSET_DIR / "run_nel_eval.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert "set -x" not in script.read_text()
