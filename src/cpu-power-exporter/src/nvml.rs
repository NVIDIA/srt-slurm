// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Superchip module power via dynamic loading of libnvidia-ml.so.
//!
//! GB200 GPUs in one socket report the same whole-module meter. The meter
//! scope and PCI NUMA-to-socket mapping were measured on GB200 only; see
//! `docs/cpu-power-telemetry.md` for the evidence. GB300 is unverified.
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

/// NVML_FI_DEV_POWER_AVERAGE: power averaged over 1 s. The instant field
/// (186) is deliberately not a fallback, so one series never mixes windows.
const POWER_FIELD_AVERAGE: u32 = 185;

/// Allows the <2% spread measured on GB200. Agreement under equal GPU load
/// cannot establish the meter's scope; that rests on the unequal-load
/// measurement in `docs/cpu-power-telemetry.md`.
const MODULE_SPREAD_TOLERANCE: f64 = 0.05;

/// During a load step a socket's GPUs can report the shared meter further
/// apart than the tolerance. In a CI sweep on GB200 and GB300, 592 of 593 such
/// disagreements lasted 3 reads or fewer. A longer one is withheld from its 4th
/// read, even when a load phase-locked to the poll keeps a valid meter
/// disagreeing.
const MAX_TRANSIENT_DISAGREEMENT_READS: u32 = 3;

/// NVML_POWER_SCOPE_MODULE. Scope 0 (NVML_POWER_SCOPE_GPU) is the GPU chip alone.
const SCOPE_MODULE: u32 = 1;

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
    fn module_power_average() -> Self {
        Self {
            field_id: POWER_FIELD_AVERAGE,
            scope_id: SCOPE_MODULE,
            _timestamp: 0,
            _latency_usec: 0,
            value_type: 0,
            nvml_return: 0,
            value: Value { ulonglong: 0 },
        }
    }

    fn watts(&self) -> Option<f64> {
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
        (mw.is_finite() && mw > 0.0).then_some(mw / 1000.0)
    }
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

struct GpuSample {
    gpu: u32,
    socket: u32,
    watts: f64,
}

#[derive(Debug, PartialEq)]
struct Disagreement {
    reading: ModuleReading,
    watts: Vec<f64>,
}

#[derive(Default)]
struct SocketState {
    mapped_gpus: usize,
    /// Consecutive disagreeing reads since all of the socket's GPUs last agreed.
    disagreeing: u32,
}

impl SocketState {
    fn withheld(&self) -> bool {
        self.disagreeing > MAX_TRANSIENT_DISAGREEMENT_READS
    }
}

pub struct NvmlModuleReader {
    session: Session,
    gpus: Vec<Gpu>,
    sockets: BTreeMap<u32, SocketState>,
}

impl NvmlModuleReader {
    pub fn new(pci_root: &Path) -> Result<Self, NvmlUnavailable> {
        let session = Session::open()?;
        let gpus = map_gpus(&session, pci_root)?;
        if gpus.is_empty() {
            return Err(NvmlUnavailable("no GPU maps to a NUMA socket".into()));
        }
        let mut reader = Self {
            session,
            sockets: socket_states(gpus.iter().map(|gpu| gpu.socket)),
            gpus,
        };
        if reader.read().is_empty() {
            return Err(NvmlUnavailable("no socket reports module power".into()));
        }
        Ok(reader)
    }

    pub fn gpu_count(&self) -> usize {
        self.gpus.len()
    }

    pub fn sockets(&self) -> Vec<u32> {
        self.sockets.keys().copied().collect()
    }

    pub fn read(&mut self) -> Vec<ModuleReading> {
        let samples = self.sample();
        publish(&mut self.sockets, &samples)
    }

    fn sample(&self) -> Vec<GpuSample> {
        let mut samples = Vec::with_capacity(self.gpus.len());
        for gpu in &self.gpus {
            let mut value = FieldValue::module_power_average();
            let ret = self
                .session
                .field_values(gpu.handle, std::slice::from_mut(&mut value));
            let watts = if ret == NVML_SUCCESS {
                value.watts()
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
        samples
    }
}

fn socket_states(gpu_sockets: impl IntoIterator<Item = u32>) -> BTreeMap<u32, SocketState> {
    let mut sockets: BTreeMap<u32, SocketState> = BTreeMap::new();
    for socket in gpu_sockets {
        sockets.entry(socket).or_default().mapped_gpus += 1;
    }
    sockets
}

/// Publish each agreeing socket, and a disagreeing one's mean until the
/// disagreement outlasts a load step.
fn publish(sockets: &mut BTreeMap<u32, SocketState>, samples: &[GpuSample]) -> Vec<ModuleReading> {
    let (readings, disagreements) = group_by_socket(samples);
    let mut published = Vec::new();
    for reading in readings {
        let Some(state) = sockets.get_mut(&reading.socket) else {
            continue;
        };
        // Only a read of all the socket's GPUs can show that they agree again.
        if reading.gpus.len() == state.mapped_gpus && state.disagreeing > 0 {
            tracing::info!(
                socket = reading.socket,
                "GPUs agree on their module power again"
            );
            state.disagreeing = 0;
        }
        if !state.withheld() {
            published.push(reading);
        }
    }
    for Disagreement { reading, watts } in disagreements {
        let Some(state) = sockets.get_mut(&reading.socket) else {
            continue;
        };
        state.disagreeing += 1;
        if !state.withheld() {
            if state.disagreeing == 1 {
                tracing::info!(
                    socket = reading.socket,
                    gpus = ?reading.gpus,
                    ?watts,
                    "GPUs disagree on their module power; publishing their mean"
                );
            }
            published.push(reading);
        } else if state.disagreeing == MAX_TRANSIENT_DISAGREEMENT_READS + 1 {
            tracing::warn!(
                socket = reading.socket,
                gpus = ?reading.gpus,
                ?watts,
                "GPUs disagree on their module power; skipping socket until they agree"
            );
        }
    }
    published.sort_by_key(|reading| reading.socket);
    published
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

/// Average the shared meter: summing would count the module once per GPU.
fn group_by_socket(samples: &[GpuSample]) -> (Vec<ModuleReading>, Vec<Disagreement>) {
    let mut by_socket: BTreeMap<u32, (Vec<u32>, Vec<f64>)> = BTreeMap::new();
    for sample in samples {
        let (gpus, watts) = by_socket.entry(sample.socket).or_default();
        gpus.push(sample.gpu);
        watts.push(sample.watts);
    }
    let mut readings = Vec::new();
    let mut disagreements = Vec::new();
    for (socket, (mut gpus, watts)) in by_socket {
        gpus.sort_unstable();
        let mean = watts.iter().sum::<f64>() / watts.len() as f64;
        let spread = watts.iter().copied().fold(f64::MIN, f64::max)
            - watts.iter().copied().fold(f64::MAX, f64::min);
        let reading = ModuleReading {
            socket,
            gpus,
            watts: mean,
        };
        if spread > MODULE_SPREAD_TOLERANCE * mean {
            disagreements.push(Disagreement { reading, watts });
        } else {
            readings.push(reading);
        }
    }
    (readings, disagreements)
}

pub fn render_module_metrics(readings: &[ModuleReading]) -> String {
    let mut out = String::from(
        "# HELP cpu_power_nvml_watts Superchip module power per Grace socket via NVML (W, 1 s average): \
         Grace, its GPUs and the rest of the module.\n\
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
    use crate::test_support::capture_logs;
    use tempfile::TempDir;

    // GPUs 0 and 1 share socket 0 and GPUs 2 and 3 socket 1, as on GB200. A
    // split's mean differs from its socket's agreeing reading, so the tests
    // see which one was published.
    const AGREE_0: &[(u32, f64)] = &[(0, 600.0), (1, 600.0)];
    const SPLIT_0: &[(u32, f64)] = &[(0, 550.0), (1, 750.0)];
    const AGREE_1: &[(u32, f64)] = &[(2, 610.0), (3, 610.0)];
    const SPLIT_1: &[(u32, f64)] = &[(2, 560.0), (3, 760.0)];

    const SPLIT_0_PUBLISHED: &str = "INFO GPUs disagree on their module power; publishing their mean socket=0 gpus=[0, 1] watts=[550.0, 750.0]";
    const SPLIT_0_WITHHELD: &str = "WARN GPUs disagree on their module power; skipping socket until they agree socket=0 gpus=[0, 1] watts=[550.0, 750.0]";
    const SOCKET_0_AGREES: &str = "INFO GPUs agree on their module power again socket=0";

    fn poll(parts: &[&[(u32, f64)]]) -> Vec<GpuSample> {
        parts
            .concat()
            .into_iter()
            .map(|(gpu, watts)| GpuSample {
                gpu,
                socket: gpu / 2,
                watts,
            })
            .collect()
    }

    fn socket_0(watts: f64) -> ModuleReading {
        ModuleReading {
            socket: 0,
            gpus: vec![0, 1],
            watts,
        }
    }

    fn socket_1(watts: f64) -> ModuleReading {
        ModuleReading {
            socket: 1,
            gpus: vec![2, 3],
            watts,
        }
    }

    fn publish_polls(polls: &[(Vec<GpuSample>, Vec<ModuleReading>)]) -> Vec<String> {
        let logs = capture_logs(|| {
            let mut sockets = socket_states([0, 0, 1, 1]);
            for (index, (samples, expected)) in polls.iter().enumerate() {
                assert_eq!(&publish(&mut sockets, samples), expected, "poll {index}");
            }
        });
        logs.lines().map(|line| line.trim().to_owned()).collect()
    }

    #[test]
    fn a_disagreeing_socket_is_published_until_the_disagreement_persists() {
        let agree = || poll(&[AGREE_0, AGREE_1]);
        let split = || poll(&[SPLIT_0, AGREE_1]);
        let agreed = || vec![socket_0(600.0), socket_1(610.0)];
        let averaged = || vec![socket_0(650.0), socket_1(610.0)];
        let withheld = || vec![socket_1(610.0)];
        let logs = publish_polls(&[
            // Even a first read that disagrees is published.
            (split(), averaged()),
            (split(), averaged()),
            (agree(), agreed()),
            // The agreement reset the count: 3 splits publish, the 4th is withheld.
            (split(), averaged()),
            (split(), averaged()),
            (split(), averaged()),
            (split(), withheld()),
            (split(), withheld()),
            (agree(), agreed()),
        ]);
        assert_eq!(
            logs,
            [
                SPLIT_0_PUBLISHED,
                SOCKET_0_AGREES,
                SPLIT_0_PUBLISHED,
                SPLIT_0_WITHHELD,
                SOCKET_0_AGREES,
            ]
        );
    }

    #[test]
    fn a_lone_gpu_reading_neither_counts_nor_ends_a_disagreement() {
        let split = || poll(&[SPLIT_0, AGREE_1]);
        let lone = || poll(&[&[(1, 620.0)], AGREE_1]);
        let lone_published = || {
            vec![
                ModuleReading {
                    socket: 0,
                    gpus: vec![1],
                    watts: 620.0,
                },
                socket_1(610.0),
            ]
        };
        let agreed = || vec![socket_0(600.0), socket_1(610.0)];
        let averaged = || vec![socket_0(650.0), socket_1(610.0)];
        let only_1 = || vec![socket_1(610.0)];
        let logs = publish_polls(&[
            // A GPU that has never reported leaves its partner published alone.
            (lone(), lone_published()),
            (split(), averaged()),
            // Neither the lone nor the absent read counts, so the next two
            // splits are the 2nd and 3rd.
            (lone(), lone_published()),
            (poll(&[AGREE_1]), only_1()),
            (split(), averaged()),
            (split(), averaged()),
            (split(), only_1()),
            // Nor does a lone reading end the withholding.
            (lone(), only_1()),
            (poll(&[AGREE_0, AGREE_1]), agreed()),
        ]);
        assert_eq!(logs, [SPLIT_0_PUBLISHED, SPLIT_0_WITHHELD, SOCKET_0_AGREES]);
    }

    #[test]
    fn sockets_disagree_independently() {
        let both_split = || poll(&[SPLIT_0, SPLIT_1]);
        let both_averaged = || vec![socket_0(650.0), socket_1(660.0)];
        let logs = publish_polls(&[
            (both_split(), both_averaged()),
            (
                poll(&[SPLIT_0, AGREE_1]),
                vec![socket_0(650.0), socket_1(610.0)],
            ),
            (both_split(), both_averaged()),
            (both_split(), vec![socket_1(660.0)]),
        ]);
        let split_1_published = "INFO GPUs disagree on their module power; publishing their mean socket=1 gpus=[2, 3] watts=[560.0, 760.0]";
        assert_eq!(
            logs,
            [
                SPLIT_0_PUBLISHED,
                split_1_published,
                "INFO GPUs agree on their module power again socket=1",
                split_1_published,
                SPLIT_0_WITHHELD,
            ]
        );
    }

    const HEADER: &str = "# HELP cpu_power_nvml_watts Superchip module power per Grace socket via NVML (W, 1 s average): Grace, its GPUs and the rest of the module.\n# TYPE cpu_power_nvml_watts gauge\n";

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
            sample(3, 1, 610.0),
            sample(0, 0, 500.0),
            sample(2, 1, 610.0),
            sample(1, 0, 498.0),
        ];
        assert_eq!(
            group_by_socket(&samples),
            (
                vec![
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
                ],
                vec![]
            )
        );
    }

    #[test]
    fn group_by_socket_separates_a_socket_whose_gpus_disagree() {
        let sample = |gpu, socket, watts| GpuSample { gpu, socket, watts };
        let samples = [
            sample(1, 0, 540.0),
            sample(0, 0, 730.0),
            sample(2, 1, 536.0),
            sample(3, 1, 533.0),
        ];
        assert_eq!(
            group_by_socket(&samples),
            (
                vec![ModuleReading {
                    socket: 1,
                    gpus: vec![2, 3],
                    watts: 534.5,
                }],
                vec![Disagreement {
                    reading: ModuleReading {
                        socket: 0,
                        gpus: vec![0, 1],
                        watts: 635.0,
                    },
                    watts: vec![540.0, 730.0],
                }]
            )
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
    fn watts_requires_nvml_success_and_a_positive_reading() {
        let filled = |nvml_return, value_type, value| FieldValue {
            nvml_return,
            value_type,
            value,
            ..FieldValue::module_power_average()
        };

        assert_eq!(
            filled(NVML_SUCCESS, VALUE_UNSIGNED_INT, Value { uint: 498_720 }).watts(),
            Some(498.72)
        );
        assert_eq!(
            filled(NVML_SUCCESS, VALUE_DOUBLE, Value { dbl: 498_720.0 }).watts(),
            Some(498.72)
        );
        assert_eq!(
            filled(
                NVML_ERROR_NOT_SUPPORTED,
                VALUE_UNSIGNED_INT,
                Value { uint: 999_999 }
            )
            .watts(),
            None
        );
        assert_eq!(
            filled(NVML_SUCCESS, VALUE_UNSIGNED_INT, Value { uint: 0 }).watts(),
            None
        );
    }

    #[test]
    fn new_fails_softly_without_nvml_or_a_socket_mapping() {
        match NvmlModuleReader::new(Path::new("/nonexistent")) {
            Ok(_) => panic!("a reader without socket-mapped GPUs must not initialise"),
            Err(e) => assert!(!e.0.is_empty(), "callers log this reason"),
        }
    }
}
