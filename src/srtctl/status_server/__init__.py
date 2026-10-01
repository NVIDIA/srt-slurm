# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native status collector: a local server for the ``reporting.status`` contract.

``srtctl status-server`` runs it; ``srtctl.status_server.store`` is the SQLite
side and ``srtctl.status_server.server`` the HTTP side. The wire shape is the
one in ``srtctl.contract`` and ``docs/status-api-spec.md``.
"""

from srtctl.status_server.server import make_server, serve
from srtctl.status_server.store import StatusStore

__all__ = ["StatusStore", "make_server", "serve"]
