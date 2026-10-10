// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Superchip module power via dynamic loading of libnvidia-ml.so.
//!
//! GPUs are grouped by the NUMA node of their PCI device, which is the Grace
//! socket they share a module with.
//!
//! All symbols are resolved at runtime so the binary starts on hosts without
//! the NVIDIA driver; unavailability is a soft `NvmlUnavailable` error.

use std::collections::BTreeMap;
use std::ffi::{c_void, CStr};
use std::fmt::{self, Write as _};
use std::fs;
use std::os::raw::c_ulong;
use std::path::Path;

// ── Constants derived from nvml.h ────────────────────────────────────────────

/// NVML_FI_DEV_POWER_AVERAGE then NVML_FI_DEV_POWER_INSTANT, in preference order.
const POWER_FIELDS: [u32; 2] = [185, 186];

/// NVML_POWER_SCOPE_MODULE. Scope 0 (NVML_POWER_SCOPE_GPU) is the GPU chip alone.
const SCOPE_MODULE: u32 = 1;

/// NVML_SUCCESS: success return code.
const NVML_SUCCESS: u32 = 0;

/// nvmlValueType_t members that can carry a power reading.
const VALUE_DOUBLE: u32 = 0;
const VALUE_UNSIGNED_INT: u32 = 1;
const VALUE_UNSIGNED_LONG: u32 = 2;
const VALUE_UNSIGNED_LONG_LONG: u32 = 3;

// ── C ABI types (nvml.h) ─────────────────────────────────────────────────────

/// nvmlDevice_t: an opaque pointer, valid until nvmlShutdown.
type Device = *mut c_void;

/// nvmlFieldValue_t, 40 bytes; size checked with ctypes against driver 580.126.20.
#[repr(C)]
struct FieldValue {
    field_id: u32,
    scope_id: u32,
    _timestamp: i64,
    _latency_usec: i64,
    value_type: u32,
    nvml_return: u32,
    value: Value,
}

/// nvmlValue_t, reduced to the members a power field can use. The omitted
/// signed members are no wider than 8 bytes, so the layout is unchanged.
#[repr(C)]
#[derive(Clone, Copy)]
union Value {
    dbl: f64,
    uint: u32,
    ulong: c_ulong,
    ulonglong: u64,
}

/// nvmlPciInfo_t as filled by nvmlDeviceGetPciInfo_v3.
#[repr(C)]
#[derive(Default)]
struct PciInfo {
    /// `0008:01:00.0`: the 4-digit domain form sysfs uses, in uppercase hex.
    bus_id_legacy: [u8; 16],
    _domain: u32,
    _bus: u32,
    _device: u32,
    _pci_device_id: u32,
    _pci_sub_system_id: u32,
    _bus_id: [u8; 32],
}

type FnInit = unsafe extern "C" fn() -> u32;
type FnShutdown = unsafe extern "C" fn() -> u32;
type FnDeviceGetCount = unsafe extern "C" fn(count: *mut u32) -> u32;
type FnDeviceGetHandleByIndex = unsafe extern "C" fn(index: u32, device: *mut Device) -> u32;
type FnDeviceGetPciInfo = unsafe extern "C" fn(device: Device, pci: *mut PciInfo) -> u32;
type FnDeviceGetFieldValues =
    unsafe extern "C" fn(device: Device, values_count: i32, values: *mut FieldValue) -> u32;

impl FieldValue {
    fn request(field_id: u32) -> Self {
        Self {
            field_id,
            scope_id: SCOPE_MODULE,
            _timestamp: 0,
            _latency_usec: 0,
            value_type: 0,
            nvml_return: 0,
            value: Value { ulonglong: 0 },
        }
    }

    fn milliwatts(&self) -> Option<f64> {
        if self.nvml_return != NVML_SUCCESS {
            return None;
        }
        // SAFETY: `value_type` names the union member NVML wrote, and every
        // member is plain data over initialised bytes.
        let mw = unsafe {
            match self.value_type {
                VALUE_DOUBLE => self.value.dbl,
                VALUE_UNSIGNED_INT => f64::from(self.value.uint),
                VALUE_UNSIGNED_LONG => self.value.ulong as f64,
                VALUE_UNSIGNED_LONG_LONG => self.value.ulonglong as f64,
                _ => return None,
            }
        };
        // Zero is no reading, as in the DCGM path: a powered module never draws 0 W.
        (mw.is_finite() && mw > 0.0).then_some(mw)
    }
}

fn module_watts(values: &[FieldValue]) -> Option<f64> {
    values
        .iter()
        .find_map(FieldValue::milliwatts)
        .map(|mw| mw / 1000.0)
}

// ── Loaded library ───────────────────────────────────────────────────────────

struct NvmlLib {
    _lib: libloading::Library,
    init: FnInit,
    shutdown: FnShutdown,
    device_get_count: FnDeviceGetCount,
    device_get_handle_by_index: FnDeviceGetHandleByIndex,
    device_get_pci_info: FnDeviceGetPciInfo,
    device_get_field_values: FnDeviceGetFieldValues,
}

impl NvmlLib {
    fn load() -> Result<Self, NvmlUnavailable> {
        let lib = Self::open_lib()?;
        // SAFETY: `_lib` lives as long as this struct, keeping each copied fn
        // pointer valid; the signatures match nvml.h.
        unsafe {
            macro_rules! sym {
                ($name:literal, $ty:ty) => {
                    *lib.get::<$ty>($name).map_err(|e| {
                        NvmlUnavailable(format!("symbol {}: {e}", stringify!($name)))
                    })?
                };
            }
            Ok(Self {
                init: sym!(b"nvmlInit_v2\0", FnInit),
                shutdown: sym!(b"nvmlShutdown\0", FnShutdown),
                device_get_count: sym!(b"nvmlDeviceGetCount_v2\0", FnDeviceGetCount),
                device_get_handle_by_index: sym!(
                    b"nvmlDeviceGetHandleByIndex_v2\0",
                    FnDeviceGetHandleByIndex
                ),
                device_get_pci_info: sym!(b"nvmlDeviceGetPciInfo_v3\0", FnDeviceGetPciInfo),
                device_get_field_values: sym!(
                    b"nvmlDeviceGetFieldValues\0",
                    FnDeviceGetFieldValues
                ),
                _lib: lib,
            })
        }
    }

    fn open_lib() -> Result<libloading::Library, NvmlUnavailable> {
        let mut last_err = String::new();
        for name in ["libnvidia-ml.so.1", "libnvidia-ml.so"] {
            match unsafe { libloading::Library::new(name) } {
                Ok(lib) => return Ok(lib),
                Err(e) => last_err = e.to_string(),
            }
        }
        Err(NvmlUnavailable(format!(
            "libnvidia-ml not found: {last_err}"
        )))
    }
}

/// A live NVML session: `nvmlInit_v2` on open, `nvmlShutdown` on drop.
struct Session(NvmlLib);

impl Session {
    fn open() -> Result<Self, NvmlUnavailable> {
        let lib = NvmlLib::load()?;
        check(unsafe { (lib.init)() }, "nvmlInit_v2")?;
        Ok(Self(lib))
    }

    fn device_count(&self) -> Result<u32, NvmlUnavailable> {
        let mut count = 0u32;
        check(
            unsafe { (self.0.device_get_count)(&mut count) },
            "nvmlDeviceGetCount_v2",
        )?;
        Ok(count)
    }

    fn locate(&self, index: u32) -> Result<(Device, String), NvmlUnavailable> {
        let mut handle: Device = std::ptr::null_mut();
        check(
            unsafe { (self.0.device_get_handle_by_index)(index, &mut handle) },
            "nvmlDeviceGetHandleByIndex_v2",
        )?;
        let mut pci = PciInfo::default();
        check(
            unsafe { (self.0.device_get_pci_info)(handle, &mut pci) },
            "nvmlDeviceGetPciInfo_v3",
        )?;
        let bus_id = CStr::from_bytes_until_nul(&pci.bus_id_legacy)
            .ok()
            .and_then(|s| s.to_str().ok())
            .ok_or_else(|| NvmlUnavailable("PCI bus ID is not a C string".into()))?;
        Ok((handle, bus_id.to_owned()))
    }

    fn field_values(&self, device: Device, values: &mut [FieldValue]) -> u32 {
        unsafe {
            (self.0.device_get_field_values)(device, values.len() as i32, values.as_mut_ptr())
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            (self.0.shutdown)();
        }
    }
}

// ── Error type ───────────────────────────────────────────────────────────────

/// NVML module power is not available on this host.
#[derive(Debug)]
pub struct NvmlUnavailable(pub String);

impl fmt::Display for NvmlUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NvmlUnavailable {}

fn check(ret: u32, context: &str) -> Result<(), NvmlUnavailable> {
    if ret == NVML_SUCCESS {
        Ok(())
    } else {
        Err(NvmlUnavailable(format!("{context}: NVML error {ret}")))
    }
}

// ── Reader ───────────────────────────────────────────────────────────────────

/// One superchip's module power, attributed to its Grace socket.
#[derive(Debug, PartialEq)]
pub struct ModuleReading {
    pub socket: u32,
    pub gpus: Vec<u32>,
    pub watts: f64,
}

struct Gpu {
    index: u32,
    socket: u32,
    handle: Device,
}

/// One GPU's module reading, before the socket's GPUs are averaged.
struct GpuSample {
    gpu: u32,
    socket: u32,
    watts: f64,
}

pub struct NvmlModuleReader {
    session: Session,
    gpus: Vec<Gpu>,
}

// SAFETY: NVML is thread-safe, and device handles stay valid until
// nvmlShutdown, which only `Session::drop` calls.
unsafe impl Send for NvmlModuleReader {}
unsafe impl Sync for NvmlModuleReader {}

impl NvmlModuleReader {
    pub fn new(pci_root: &Path) -> Result<Self, NvmlUnavailable> {
        let session = Session::open()?;
        let gpus = map_gpus(&session, pci_root)?;
        if gpus.is_empty() {
            return Err(NvmlUnavailable("no GPU maps to a NUMA socket".into()));
        }
        let reader = Self { session, gpus };
        if reader.read().is_empty() {
            return Err(NvmlUnavailable("no GPU reports module power".into()));
        }
        Ok(reader)
    }

    pub fn gpu_count(&self) -> usize {
        self.gpus.len()
    }

    pub fn sockets(&self) -> Vec<u32> {
        let mut sockets: Vec<u32> = self.gpus.iter().map(|g| g.socket).collect();
        sockets.sort_unstable();
        sockets.dedup();
        sockets
    }

    pub fn read(&self) -> Vec<ModuleReading> {
        let mut samples = Vec::with_capacity(self.gpus.len());
        for gpu in &self.gpus {
            let mut values = POWER_FIELDS.map(FieldValue::request);
            let ret = self.session.field_values(gpu.handle, &mut values);
            let watts = if ret == NVML_SUCCESS {
                module_watts(&values)
            } else {
                None
            };
            match watts {
                Some(watts) => samples.push(GpuSample {
                    gpu: gpu.index,
                    socket: gpu.socket,
                    watts,
                }),
                None => tracing::debug!(
                    gpu = gpu.index,
                    ret,
                    "no module power reading; skipping GPU"
                ),
            }
        }
        group_by_socket(&samples)
    }
}

fn map_gpus(session: &Session, pci_root: &Path) -> Result<Vec<Gpu>, NvmlUnavailable> {
    let mut gpus = Vec::new();
    for index in 0..session.device_count()? {
        let (handle, bus_id) = match session.locate(index) {
            Ok(found) => found,
            Err(e) => {
                tracing::warn!(gpu = index, error = %e, "GPU not addressable; module power skips it");
                continue;
            }
        };
        match socket_of_bus_id(pci_root, &bus_id) {
            Some(socket) => gpus.push(Gpu {
                index,
                socket,
                handle,
            }),
            None => {
                tracing::warn!(gpu = index, %bus_id, "GPU has no known NUMA socket; module power skips it")
            }
        }
    }
    Ok(gpus)
}

// ── Pure helpers ─────────────────────────────────────────────────────────────

/// The NUMA node sysfs reports for a PCI device: None when missing,
/// unparsable, or -1 (no affinity).
fn socket_of_bus_id(pci_root: &Path, bus_id_legacy: &str) -> Option<u32> {
    // sysfs names devices in lowercase hex; NVML prints uppercase.
    let numa_node = pci_root
        .join(bus_id_legacy.to_lowercase())
        .join("numa_node");
    fs::read_to_string(numa_node).ok()?.trim().parse().ok()
}

/// One reading per socket. Every GPU on a superchip reports the same module
/// value, so the socket's reading is their mean; a sum would count the module
/// once per GPU.
fn group_by_socket(samples: &[GpuSample]) -> Vec<ModuleReading> {
    let mut by_socket: BTreeMap<u32, (Vec<u32>, f64)> = BTreeMap::new();
    for sample in samples {
        let (gpus, total) = by_socket.entry(sample.socket).or_default();
        gpus.push(sample.gpu);
        *total += sample.watts;
    }
    by_socket
        .into_iter()
        .map(|(socket, (mut gpus, total))| {
            gpus.sort_unstable();
            let watts = total / gpus.len() as f64;
            ModuleReading {
                socket,
                gpus,
                watts,
            }
        })
        .collect()
}

pub fn render_module_metrics(readings: &[ModuleReading]) -> String {
    let mut out = String::from(
        "# HELP cpu_power_nvml_watts Superchip module power per Grace socket via NVML (W): \
         Grace, its GPUs, HBM, LPDDR5X and regulators.\n\
         # TYPE cpu_power_nvml_watts gauge\n",
    );
    for r in readings {
        let gpus: Vec<String> = r.gpus.iter().map(u32::to_string).collect();
        let _ = writeln!(
            out,
            "cpu_power_nvml_watts{{type=\"module\",socket=\"{}\",gpus=\"{}\",source=\"nvml\"}} {:.6}",
            r.socket,
            gpus.join(","),
            r.watts,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const HEADER: &str = "# HELP cpu_power_nvml_watts Superchip module power per Grace socket via NVML (W): Grace, its GPUs, HBM, LPDDR5X and regulators.\n# TYPE cpu_power_nvml_watts gauge\n";

    const NVML_ERROR_NOT_SUPPORTED: u32 = 3;

    #[test]
    fn ffi_struct_layouts_match_nvml_h() {
        use std::mem::{offset_of, size_of};
        // Offsets, not just sizes: a narrowed field can hide in padding.
        assert_eq!(offset_of!(FieldValue, scope_id), 4);
        assert_eq!(offset_of!(FieldValue, value_type), 24);
        assert_eq!(offset_of!(FieldValue, nvml_return), 28);
        assert_eq!(offset_of!(FieldValue, value), 32);
        assert_eq!(size_of::<FieldValue>(), 40);
        assert_eq!(size_of::<PciInfo>(), 68);
    }

    #[test]
    fn socket_of_bus_id_reads_the_pci_numa_node() {
        let root = TempDir::new().unwrap();
        for (dir, numa_node) in [
            ("0008:01:00.0", "0\n"),
            ("0000:1b:00.0", "1\n"),
            ("0009:01:00.0", "-1\n"),
            ("0018:01:00.0", "socket0\n"),
        ] {
            fs::create_dir_all(root.path().join(dir)).unwrap();
            fs::write(root.path().join(dir).join("numa_node"), numa_node).unwrap();
        }
        assert_eq!(socket_of_bus_id(root.path(), "0008:01:00.0"), Some(0));
        assert_eq!(socket_of_bus_id(root.path(), "0000:1B:00.0"), Some(1));
        assert_eq!(socket_of_bus_id(root.path(), "0009:01:00.0"), None);
        assert_eq!(socket_of_bus_id(root.path(), "0018:01:00.0"), None);
        assert_eq!(socket_of_bus_id(root.path(), "0019:01:00.0"), None);
    }

    #[test]
    fn group_by_socket_averages_each_module_once() {
        let sample = |gpu, socket, watts| GpuSample { gpu, socket, watts };
        let samples = [
            sample(3, 1, 600.0),
            sample(0, 0, 500.0),
            sample(2, 1, 620.0),
            sample(1, 0, 498.0),
        ];
        assert_eq!(
            group_by_socket(&samples),
            [
                ModuleReading {
                    socket: 0,
                    gpus: vec![0, 1],
                    watts: 499.0,
                },
                ModuleReading {
                    socket: 1,
                    gpus: vec![2, 3],
                    watts: 610.0,
                },
            ]
        );
    }

    #[test]
    fn render_module_metrics_emits_one_series_per_socket() {
        let readings = [
            ModuleReading {
                socket: 0,
                gpus: vec![0, 1],
                watts: 498.72,
            },
            ModuleReading {
                socket: 1,
                gpus: vec![2, 3],
                watts: 501.5,
            },
        ];
        assert_eq!(
            render_module_metrics(&readings),
            format!(
                "{HEADER}\
                 cpu_power_nvml_watts{{type=\"module\",socket=\"0\",gpus=\"0,1\",source=\"nvml\"}} 498.720000\n\
                 cpu_power_nvml_watts{{type=\"module\",socket=\"1\",gpus=\"2,3\",source=\"nvml\"}} 501.500000\n"
            )
        );
        assert_eq!(render_module_metrics(&[]), HEADER);
    }

    #[test]
    fn module_watts_prefers_the_average_and_falls_back_to_instant() {
        let filled = |nvml_return, value_type, value| FieldValue {
            nvml_return,
            value_type,
            value,
            ..FieldValue::request(0)
        };
        let not_supported = || {
            filled(
                NVML_ERROR_NOT_SUPPORTED,
                VALUE_UNSIGNED_INT,
                Value { uint: 999_999 },
            )
        };
        let uint = |mw| filled(NVML_SUCCESS, VALUE_UNSIGNED_INT, Value { uint: mw });

        assert_eq!(module_watts(&[uint(498_720), uint(510_000)]), Some(498.72));
        assert_eq!(module_watts(&[not_supported(), uint(510_000)]), Some(510.0));
        assert_eq!(module_watts(&[not_supported(), not_supported()]), None);
        assert_eq!(
            module_watts(&[filled(NVML_SUCCESS, VALUE_DOUBLE, Value { dbl: 498_720.0 })]),
            Some(498.72)
        );
    }

    /// Without libnvidia-ml this fails at load; with it, at the socket mapping.
    #[test]
    fn new_fails_softly_without_nvml_or_a_socket_mapping() {
        match NvmlModuleReader::new(Path::new("/nonexistent")) {
            Ok(_) => panic!("a reader without socket-mapped GPUs must not initialise"),
            Err(e) => assert!(!e.0.is_empty(), "callers log this reason"),
        }
    }
}
