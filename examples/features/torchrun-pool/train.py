# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every rank joins one process group, proves the others are there, and says hello.

Launched by torchrun from examples/features/torchrun-pool.yaml: one torchrun per
node of the pool, one process per GPU. torchrun sets RANK, LOCAL_RANK, WORLD_SIZE,
MASTER_ADDR and MASTER_PORT; init_process_group reads them from the environment.
"""

import os
import socket
import sys

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Every rank contributes a one; the sum is how many ranks actually joined.
    joined = torch.ones(1, device="cuda")
    dist.all_reduce(joined)

    # One write per rank: eight ranks share this node's stdout, and separate writes
    # for the text and the newline interleave into merged lines.
    line = (
        f"from worker {rank} srt-slurm is just better kubernetes "
        f"({int(joined.item())}/{world} joined, {socket.gethostname()} gpu {local_rank})\n"
    )
    sys.stdout.write(line)
    sys.stdout.flush()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
