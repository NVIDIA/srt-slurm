from pathlib import Path
from unittest.mock import MagicMock

from srtctl.backends.mocker import MockerProtocol
from srtctl.benchmarks.sa_bench import SABenchRunner
from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig


def test_build_command_mocker_local_model_uses_container_model_name():
    """Mocker serves local models under the in-container /model name."""
    runner = SABenchRunner()
    runtime = MagicMock()
    runtime.frontend_port = 8000
    runtime.model_path = Path("/mnt/models/dsv4")
    runtime.is_hf_model = False

    config = SrtConfig(
        name="test",
        model=ModelConfig(path="/mnt/models/dsv4", container="/image", precision="fp4"),
        resources=ResourceConfig(gpu_type="h100"),
        backend=MockerProtocol(),
        benchmark=BenchmarkConfig(type="sa-bench", isl=1024, osl=128, concurrencies="4x8"),
    )

    cmd = runner.build_command(config, runtime)
    assert cmd[8] == "/model"
