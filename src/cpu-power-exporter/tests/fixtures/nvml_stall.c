// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

unsigned int nvmlInit_v2(void) {
    FILE *marker = fopen(getenv("NVML_TEST_ENTERED"), "w");
    if (marker != NULL) {
        fclose(marker);
    }
    for (;;) {
        sleep(60);
    }
}

unsigned int nvmlShutdown(void) { return 0; }
unsigned int nvmlDeviceGetCount_v2(unsigned int *count) {
    *count = 0;
    return 0;
}
unsigned int nvmlDeviceGetHandleByIndex_v2(unsigned int index, void **device) {
    return 999;
}
unsigned int nvmlDeviceGetPciInfo_v3(void *device, void *pci) { return 999; }
unsigned int nvmlDeviceGetFieldValues(void *device, int count, void *values) {
    return 999;
}
