"""
Patch vLLM's CuMemAllocator to be compatible with PyTorch expandable
segments by temporarily toggling the allocator setting around the memory
pool context (sleep mode), instead of hard-asserting at __init__ time.

Backports vllm-project/vllm#40812 ("Auto-disable expandable_segments
around cumem memory pool"). Without this patch, setting
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True together with
enable-sleep-mode causes vLLM to abort during CuMemAllocator
construction; with this patch, expandable segments stay on for normal
allocations and are flipped off only for the duration of
use_memory_pool().

Reference: https://github.com/vllm-project/vllm/pull/40812
Affected file: vllm/device_allocator/cumem.py
"""

import sys
from pathlib import Path

TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/device_allocator/cumem.py")

# Idempotency: the new use_memory_pool body introduces this exact line.
MARKER = 'expandable_was_enabled = "expandable_segments:True" in conf'

# --- Hunk 1: drop the __init__ assertion -------------------------------------

INIT_OLD = (
    "    def __init__(self):\n"
    '        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")\n'
    '        assert "expandable_segments:True" not in conf, (\n'
    '            "Expandable segments are not compatible with memory pool. "\n'
    '            "Please track https://github.com/pytorch/pytorch/issues/147851 "\n'
    '            "for the latest updates."\n'
    "        )\n"
    "\n"
    "        self.pointer_to_data: dict[int, AllocationData] = {}\n"
)

INIT_NEW = (
    "    def __init__(self):\n"
    "        self.pointer_to_data: dict[int, AllocationData] = {}\n"
)

# --- Hunk 2: wrap use_memory_pool body in try/finally + toggle ---------------

POOL_OLD = (
    "        assert isinstance(tag, str)\n"
    "\n"
    "        old_tag = self.current_tag\n"
    "        self.current_tag = tag\n"
    "        with use_memory_pool_with_allocator(\n"
    "            self.python_malloc_callback, self.python_free_callback\n"
    "        ) as data:\n"
    "            # start to hit another PyTorch bug in PyTorch 2.6,\n"
    "            # possibly because of gc-related issue w.r.t. the allocator and\n"
    "            # the memory pool.\n"
    "            # to avoid the issue, we keep a reference of the data.\n"
    "            # see https://github.com/pytorch/pytorch/issues/146431 .\n"
    "            self.allocator_and_pools[tag] = data\n"
    "            yield\n"
    "            # PyTorch's bug, calling torch.cuda.empty_cache() will error\n"
    "            # when using pluggable allocator, see\n"
    "            # https://github.com/pytorch/pytorch/issues/145168 .\n"
    "            # if we have some memory allocated and then freed,\n"
    "            # the memory will not be released, e.g. in online quantization,\n"
    "            # where the model is created in higher precision, and then\n"
    "            # quantized in lower precision.\n"
    "            # Find all unused allocations and manually release them.\n"
    "            # TODO: we should expose `empty_cache` method in the memory pool.\n"
    "            # TODO: ask for help from PyTorch team to expose this method.\n"
    "            allocations = data[0].snapshot()\n"
    "            for allocation in allocations:\n"
    "                if allocation[\"allocated_size\"] == 0:\n"
    "                    handle = self._python_free_callback(allocation[\"address\"])\n"
    "                    unmap_and_release(handle)\n"
    "            self.current_tag = old_tag\n"
)

POOL_NEW = (
    "        assert isinstance(tag, str)\n"
    "\n"
    "        # Expandable segments are incompatible with the memory pool used for\n"
    "        # sleep mode (see https://github.com/pytorch/pytorch/issues/147851).\n"
    "        # If the user has enabled expandable segments via\n"
    "        # PYTORCH_CUDA_ALLOC_CONF, temporarily disable them for the duration\n"
    "        # of the memory pool context and restore on exit.\n"
    '        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")\n'
    '        expandable_was_enabled = "expandable_segments:True" in conf\n'
    "        if expandable_was_enabled:\n"
    '            torch.cuda.memory._set_allocator_settings("expandable_segments:False")\n'
    "\n"
    "        old_tag = self.current_tag\n"
    "        self.current_tag = tag\n"
    "        try:\n"
    "            with use_memory_pool_with_allocator(\n"
    "                self.python_malloc_callback, self.python_free_callback\n"
    "            ) as data:\n"
    "                # start to hit another PyTorch bug in PyTorch 2.6,\n"
    "                # possibly because of gc-related issue w.r.t. the allocator\n"
    "                # and the memory pool.\n"
    "                # to avoid the issue, we keep a reference of the data.\n"
    "                # see https://github.com/pytorch/pytorch/issues/146431 .\n"
    "                self.allocator_and_pools[tag] = data\n"
    "                yield\n"
    "                # PyTorch's bug, calling torch.cuda.empty_cache() will error\n"
    "                # when using pluggable allocator, see\n"
    "                # https://github.com/pytorch/pytorch/issues/145168 .\n"
    "                # if we have some memory allocated and then freed,\n"
    "                # the memory will not be released, e.g. in online\n"
    "                # quantization, where the model is created in higher\n"
    "                # precision, and then quantized in lower precision.\n"
    "                # Find all unused allocations and manually release them.\n"
    "                # TODO: we should expose `empty_cache` method in the memory\n"
    "                # pool.\n"
    "                # TODO: ask for help from PyTorch team to expose this method.\n"
    "                allocations = data[0].snapshot()\n"
    "                for allocation in allocations:\n"
    "                    if allocation[\"allocated_size\"] == 0:\n"
    "                        handle = self._python_free_callback(allocation[\"address\"])\n"
    "                        unmap_and_release(handle)\n"
    "        finally:\n"
    "            self.current_tag = old_tag\n"
    "            if expandable_was_enabled:\n"
    '                torch.cuda.memory._set_allocator_settings("expandable_segments:True")\n'
)

PATCHES = [
    ("CuMemAllocator.__init__ assertion removal", INIT_OLD, INIT_NEW),
    ("CuMemAllocator.use_memory_pool toggle", POOL_OLD, POOL_NEW),
]


def main():
    if not TARGET.exists():
        print(f"[vllm-cumem-expandable-fix] Target not found: {TARGET}", file=sys.stderr)
        sys.exit(1)

    content = TARGET.read_text()
    if MARKER in content:
        print("[vllm-cumem-expandable-fix] Already patched, skipping.", file=sys.stderr)
        return

    new_content = content
    for name, old, new in PATCHES:
        count = new_content.count(old)
        if count == 0:
            print(
                f"[vllm-cumem-expandable-fix] Anchor for {name!r} not found. "
                "vLLM version may have drifted; inspect cumem.py.",
                file=sys.stderr,
            )
            sys.exit(1)
        if count > 1:
            print(
                f"[vllm-cumem-expandable-fix] Anchor for {name!r} is ambiguous "
                f"({count} matches); refusing to patch.",
                file=sys.stderr,
            )
            sys.exit(1)
        new_content = new_content.replace(old, new, 1)
        print(f"[vllm-cumem-expandable-fix] Patched {name}", file=sys.stderr)

    TARGET.write_text(new_content)
    print("[vllm-cumem-expandable-fix] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
