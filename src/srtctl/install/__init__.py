# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot installation flow for srtctl models and containers.

Exposes the building blocks consumed by ``srtctl install <model>``:

- :mod:`registry`            hardcoded model registry (beta)
- :mod:`setup`               NATS/ETCD bootstrap detection
- :mod:`model`               Hugging Face model download
- :mod:`container`           enroot ``.sqsh`` import
- :mod:`srtslurm_yaml_writer` add ``model_paths`` / ``containers`` aliases
"""
