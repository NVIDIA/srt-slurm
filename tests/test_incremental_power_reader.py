# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from srtctl.analysis.incremental_power import read_csv_tolerantly


def test_complete_file_is_returned_intact(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("a,b\n1,2\n")

    assert read_csv_tolerantly(path).read() == "a,b\n1,2\n"


def test_torn_final_row_is_dropped(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("a,b\n1,2\n3,")

    assert read_csv_tolerantly(path).read() == "a,b\n1,2\n"


def test_file_with_no_newline_at_all_yields_nothing(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("a,b")

    assert read_csv_tolerantly(path).read() == ""
