// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Standalone physical-host-allocation reproducer. Compile with a normal C++17
// compiler, CUDA >= 12.9 headers, and -lcuda; no CUDA Runtime or other framework.
// Properties match the failing cuMemCreate in Mooncake commit
// 042ab8a18b0e6ccfe87782261a5556a3dd4446e4. No GPU buffers, address mapping,
// copies, kernels, memory touching, or CUDA context creation are performed.

#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#if CUDA_VERSION < 12090
#error "CUDA >= 12.9 headers are required for HOST_NUMA VMM support"
#endif

namespace {

volatile std::sig_atomic_t stop_requested = 0;

void stop_handler(int) { stop_requested = 1; }

struct Options {
    int device = 0;
    int numa = -1;
    std::uint64_t size_bytes = 0;
    std::uint64_t chunk_bytes = 0;
    std::string ready_file;
    std::string result_file;
};

std::string json_quote(const std::string& value) {
    std::ostringstream output;
    output << '"';
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch)
                           << std::dec;
                } else {
                    output << ch;
                }
        }
    }
    output << '"';
    return output.str();
}

std::uint64_t unsigned_value(const std::string& value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected an unsigned integer, got " + value);
    std::size_t consumed = 0;
    auto parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size()) throw std::invalid_argument("invalid integer: " + value);
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--help") {
            std::cout << "Usage: cuda-host-numa-repro --device VISIBLE_ORDINAL --numa NODE --size-bytes N\n"
                         "       [--chunk-bytes N] --ready-file PATH --result-file PATH\n"
                         "Allocates PINNED HOST_NUMA FABRIC handles and holds them until SIGTERM/SIGINT.\n"
                         "--chunk-bytes 0 (default) makes one allocation. Positive values cap balanced chunks.\n"
                         "Allocation failures retain successful partial handles until signaled, for inspection.\n";
            std::exit(0);
        }
        if (++index == argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[index];
        if (key == "--ready-file") options.ready_file = value;
        else if (key == "--result-file") options.result_file = value;
        else if (key == "--size-bytes") options.size_bytes = unsigned_value(value);
        else if (key == "--chunk-bytes") options.chunk_bytes = unsigned_value(value);
        else if (key == "--device" || key == "--numa") {
            const auto parsed = unsigned_value(value);
            if (parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument(key + " exceeds int range");
            if (key == "--device") options.device = static_cast<int>(parsed);
            else options.numa = static_cast<int>(parsed);
        } else throw std::invalid_argument("unknown option " + key);
    }
    if (options.numa < 0 || options.size_bytes == 0 || options.ready_file.empty() || options.result_file.empty())
        throw std::invalid_argument("--numa, positive --size-bytes, --ready-file and --result-file are required");
    if (options.size_bytes > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("size exceeds size_t range");
    if (options.ready_file == options.result_file)
        throw std::invalid_argument("ready and result files must differ");
    return options;
}

struct Call {
    std::string api;
    int result;
    std::string name;
    std::string message;
    std::uint64_t bytes;
    double elapsed_seconds;
};

struct Handle {
    CUmemGenericAllocationHandle handle;
    std::uint64_t bytes;
};

class Reproducer {
public:
    explicit Reproducer(Options options) : options_(std::move(options)) {}

    int run() {
        std::signal(SIGTERM, stop_handler);
        std::signal(SIGINT, stop_handler);
        try {
            write_result();
            setup();
        } catch (const std::exception& error) {
            error_ = error.what();
            if (status_ != "allocation_failed") status_ = "setup_failed";
            std::cerr << error_ << '\n';
        }
        allocation_status_ = status_;
        setup_complete_ = true;
        bool output_ok = recording_ok_;
        try {
            write_result();
            atomic_write(options_.ready_file, "{\"setup_complete\":true,\"status\":" + json_quote(status_) + "}\n");
        } catch (const std::exception& error) {
            std::cerr << "Cannot publish allocation state: " << error.what() << '\n';
            output_ok = false;
        }
        // Keep partial allocations alive after OOM. The supervisor records the
        // memory state before signaling us to release them in reverse order.
        while (output_ok && !stop_requested) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        bool cleanup_ok = release_handles();
        cleanup_confirmed_ = cleanup_ok && handles_.empty();
        status_ = cleanup_confirmed_ ? "closed" : "cleanup_failed";
        try {
            write_result();
        } catch (const std::exception& error) {
            std::cerr << "Cannot publish cleanup state: " << error.what() << '\n';
            output_ok = false;
        }
        if (!cleanup_ok || !output_ok) return 1;
        return allocation_success_ ? 0 : 2;
    }

private:
    Options options_;
    std::string status_ = "initializing";
    std::string allocation_status_;
    std::string current_call_;
    std::string failed_api_;
    int failed_result_ = 0;
    std::string error_;
    bool recording_ok_ = true;
    bool setup_complete_ = false;
    bool allocation_success_ = false;
    bool cleanup_confirmed_ = false;
    int driver_version_ = 0;
    int device_count_ = 0;
    int host_numa_id_ = -1;
    std::string device_name_;
    std::string device_pci_bus_id_;
    std::string device_uuid_;
    std::uint64_t granularity_ = 0;
    std::uint64_t effective_bytes_ = 0;
    std::uint64_t chunk_bytes_ = 0;
    std::uint64_t allocated_bytes_ = 0;
    std::vector<Handle> handles_;
    std::vector<Call> calls_;

    static void atomic_write(const std::string& path, const std::string& contents) {
        const auto temporary = path + ".tmp-" + std::to_string(getpid());
        {
            std::ofstream stream(temporary, std::ios::out | std::ios::trunc);
            stream.exceptions(std::ios::badbit | std::ios::failbit);
            stream << contents;
            stream.close();
        }
        if (std::rename(temporary.c_str(), path.c_str()) != 0) {
            std::remove(temporary.c_str());
            throw std::runtime_error("rename failed for " + path);
        }
    }

    void write_result() const {
        std::uint64_t held_bytes = 0;
        for (const auto& handle : handles_) held_bytes += handle.bytes;
        const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
        std::ostringstream output;
        output << "{\"schema_version\":1,\"pid\":" << getpid()
               << ",\"status\":" << json_quote(status_) << ",\"current_call\":" << json_quote(current_call_)
               << ",\"allocation_status\":" << json_quote(allocation_status_)
               << ",\"failed_api\":" << json_quote(failed_api_) << ",\"failed_result\":" << failed_result_
               << ",\"error\":" << json_quote(error_)
               << ",\"setup_complete\":" << (setup_complete_ ? "true" : "false")
               << ",\"allocation_success\":" << (allocation_success_ ? "true" : "false")
               << ",\"cleanup_confirmed\":" << (cleanup_confirmed_ ? "true" : "false")
               << ",\"device\":" << options_.device << ",\"numa\":" << options_.numa
               << ",\"requested_bytes\":" << options_.size_bytes << ",\"effective_bytes\":" << effective_bytes_
               << ",\"chunk_limit_bytes\":" << options_.chunk_bytes << ",\"chunk_bytes\":" << chunk_bytes_
               << ",\"granularity_bytes\":" << granularity_ << ",\"allocated_bytes\":" << allocated_bytes_
               << ",\"held_bytes\":" << held_bytes << ",\"live_handles\":" << handles_.size()
               << ",\"allocation_type\":\"PINNED\",\"location_type\":\"HOST_NUMA\",\"handle_type\":\"FABRIC\""
               << ",\"cuda\":{\"header_version\":" << CUDA_VERSION << ",\"driver_version\":" << driver_version_
               << ",\"visible_devices\":" << json_quote(visible ? visible : "") << ",\"device_count\":" << device_count_
               << ",\"device_name\":" << json_quote(device_name_) << ",\"pci_bus_id\":" << json_quote(device_pci_bus_id_)
               << ",\"uuid\":" << json_quote(device_uuid_) << ",\"host_numa_id\":" << host_numa_id_ << "},\"calls\":[";
        for (std::size_t index = 0; index < calls_.size(); ++index) {
            const auto& call = calls_[index];
            if (index) output << ',';
            output << "{\"api\":" << json_quote(call.api) << ",\"result\":" << call.result
                   << ",\"name\":" << json_quote(call.name) << ",\"message\":" << json_quote(call.message)
                   << ",\"bytes\":" << call.bytes << ",\"elapsed_seconds\":" << call.elapsed_seconds << '}';
        }
        output << "]}\n";
        atomic_write(options_.result_file, output.str());
    }

    void record_state() {
        try {
            write_result();
        } catch (const std::exception& error) {
            recording_ok_ = false;
            std::cerr << "Cannot record CUDA call state: " << error.what() << '\n';
        }
    }

    CUresult call(const std::string& name, const std::function<CUresult()>& function, std::uint64_t bytes = 0) {
        current_call_ = name;
        record_state();
        std::cout << name << " begin bytes=" << bytes << std::endl;
        const auto started = std::chrono::steady_clock::now();
        const CUresult result = function();
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const char* error_name = nullptr;
        const char* error_string = nullptr;
        cuGetErrorName(result, &error_name);
        cuGetErrorString(result, &error_string);
        calls_.push_back({name, static_cast<int>(result), error_name ? error_name : "UNKNOWN",
                          error_string ? error_string : "unavailable", bytes, elapsed});
        if (result != CUDA_SUCCESS && failed_api_.empty()) {
            failed_api_ = name;
            failed_result_ = static_cast<int>(result);
        }
        current_call_.clear();
        std::cout << name << " result=" << static_cast<int>(result) << " name=" << calls_.back().name
                  << " elapsed_seconds=" << elapsed << std::endl;
        record_state();
        return result;
    }

    void require(const std::string& name, const std::function<CUresult()>& function) {
        if (call(name, function) != CUDA_SUCCESS)
            throw std::runtime_error(name + " failed: " + calls_.back().name + " (" +
                                     std::to_string(calls_.back().result) + "): " + calls_.back().message);
    }

    static std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
        if (value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1))
            throw std::invalid_argument("allocation size overflows when aligned");
        return ((value + alignment - 1) / alignment) * alignment;
    }

    void setup() {
        require("cuInit", [] { return cuInit(0); });
        require("cuDriverGetVersion", [&] { return cuDriverGetVersion(&driver_version_); });
        require("cuDeviceGetCount", [&] { return cuDeviceGetCount(&device_count_); });
        if (options_.device >= device_count_) throw std::invalid_argument("visible device ordinal is unavailable");
        CUdevice device = 0;
        require("cuDeviceGet", [&] { return cuDeviceGet(&device, options_.device); });
        char name[256] = {};
        require("cuDeviceGetName", [&] { return cuDeviceGetName(name, sizeof(name), device); });
        device_name_ = name;
        char bus_id[64] = {};
        require("cuDeviceGetPCIBusId", [&] { return cuDeviceGetPCIBusId(bus_id, sizeof(bus_id), device); });
        device_pci_bus_id_ = bus_id;
        CUuuid uuid{};
        require("cuDeviceGetUuid", [&] { return cuDeviceGetUuid(&uuid, device); });
        std::ostringstream uuid_string;
        for (auto byte : uuid.bytes)
            uuid_string << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(static_cast<unsigned char>(byte));
        device_uuid_ = uuid_string.str();
        require("cuDeviceGetAttribute(HOST_NUMA_ID)", [&] {
            return cuDeviceGetAttribute(&host_numa_id_, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device);
        });
        const std::vector<std::pair<CUdevice_attribute, const char*>> capabilities = {
            {CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, "VMM"},
            {CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, "FABRIC"},
            {CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, "HOST_NUMA_VMM"},
        };
        for (const auto& capability : capabilities) {
            int supported = 0;
            require(std::string("cuDeviceGetAttribute(") + capability.second + ")", [&] {
                return cuDeviceGetAttribute(&supported, capability.first, device);
            });
            if (!supported) throw std::runtime_error(std::string("device lacks ") + capability.second);
        }
        CUmemAllocationProp properties{};
        properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        properties.location.id = options_.numa;
        properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        std::size_t minimum = 0;
        require("cuMemGetAllocationGranularity", [&] {
            return cuMemGetAllocationGranularity(&minimum, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        });
        granularity_ = minimum;
        if (minimum == 0 || (minimum & (minimum - 1)) != 0)
            throw std::runtime_error("CUDA returned invalid allocation granularity");
        effective_bytes_ = align_up(options_.size_bytes, minimum);
        if (effective_bytes_ > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("aligned size exceeds size_t range");
        chunk_bytes_ = effective_bytes_;
        if (options_.chunk_bytes) {
            const auto limit = options_.chunk_bytes - (options_.chunk_bytes % minimum);
            if (!limit) throw std::invalid_argument("chunk cap is below CUDA allocation granularity");
            const auto count = 1 + (effective_bytes_ - 1) / limit;
            if (count > 1000000) throw std::invalid_argument("chunk count exceeds 1000000");
            chunk_bytes_ = align_up(1 + (effective_bytes_ - 1) / count, minimum);
        }
        handles_.reserve(static_cast<std::size_t>(1 + (effective_bytes_ - 1) / chunk_bytes_));
        std::cout << "HOST_NUMA=" << options_.numa << " requested_bytes=" << options_.size_bytes
                  << " effective_bytes=" << effective_bytes_ << " granularity_bytes=" << granularity_
                  << " balanced_chunk_bytes=" << chunk_bytes_ << std::endl;
        status_ = "allocating";
        write_result();
        while (allocated_bytes_ < effective_bytes_ && !stop_requested) {
            const auto bytes = std::min(chunk_bytes_, effective_bytes_ - allocated_bytes_);
            CUmemGenericAllocationHandle handle{};
            const auto result = call("cuMemCreate", [&] {
                return cuMemCreate(&handle, static_cast<std::size_t>(bytes), &properties, 0);
            }, bytes);
            if (result != CUDA_SUCCESS) {
                status_ = "allocation_failed";
                throw std::runtime_error("cuMemCreate failed: " + calls_.back().name + " (" +
                                         std::to_string(calls_.back().result) + "): " + calls_.back().message);
            }
            handles_.push_back({handle, bytes});
            allocated_bytes_ += bytes;
            std::cout << "Resident handles=" << handles_.size() << " allocated_bytes=" << allocated_bytes_ << std::endl;
            write_result();
        }
        allocation_success_ = allocated_bytes_ == effective_bytes_;
        status_ = allocation_success_ ? "ready" : "interrupted";
    }

    bool release_handles() {
        bool success = true;
        for (std::size_t index = handles_.size(); index > 0; --index) {
            const auto handle = handles_[index - 1];
            try {
                if (call("cuMemRelease", [&] { return cuMemRelease(handle.handle); }, handle.bytes) == CUDA_SUCCESS)
                    handles_.erase(handles_.begin() + static_cast<std::ptrdiff_t>(index - 1));
                else success = false;
            } catch (const std::exception& error) {
                std::cerr << "cuMemRelease state recording failed: " << error.what() << '\n';
                success = false;
            }
        }
        return success && recording_ok_;
    }
};

}  // namespace

int main(int argc, char** argv) {
    try {
        return Reproducer(parse_options(argc, argv)).run();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
