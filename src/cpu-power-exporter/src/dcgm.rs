// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DCGM CPU power reader via dynamic loading of libdcgm.so.
//!
//! Field 1130 (DCGM_FI_DEV_CPU_POWER_WATTS / DCGM_FI_DEV_CPU_POWER_UTIL_CURRENT) is a watched
//! field, not a live-data field.  The correct setup sequence is:
//!   1. dcgmInit / dcgmStartEmbedded  — initialise the library and get a handle.
//!   2. dcgmGetEntityGroupEntities    — enumerate DCGM_FE_CPU entity IDs.
//!   3. dcgmGroupCreate + dcgmGroupAddEntity × N — build an entity group.
//!   4. dcgmFieldGroupCreate          — name a field group containing field 1130.
//!   5. dcgmWatchFields               — register the watch on that entity+field group.
//!   6. dcgmUpdateAllFields(true)     — seed the first sample.
//!   7. dcgmEntitiesGetLatestValues with flags=0 — read cached values on every tick.
//!
//! Passing DCGM_FV_FLAG_LIVE_DATA (0x1) to step 7 bypasses the cache and returns
//! blank/stale data for watched fields — that is the bug Kyle's c1c0028 fixed.
//!
//! All symbols are resolved at runtime so the binary starts on hosts that lack
//! DCGM; unavailability is a soft `DcgmUnavailable` error, not a link failure.

use std::ffi::CString;
use std::fmt;

// ── Constants derived from dcgm_fields.h / dcgm_structs.h ────────────────────

/// DCGM_FI_DEV_CPU_POWER_WATTS (alias DCGM_FI_DEV_CPU_POWER_UTIL_CURRENT).
const CPU_POWER_FIELD_ID: u16 = 1130;

/// DCGM_FE_CPU: entity group for Grace CPU nodes.
const DCGM_FE_CPU: u32 = 7;

/// DCGM_GROUP_EMPTY: create an empty entity group.
const DCGM_GROUP_EMPTY: i32 = 1;

/// DCGM_OPERATION_MODE_AUTO: collect data automatically.
const DCGM_OPERATION_MODE_AUTO: i32 = 1;

/// DCGM_GEGE_FLAG_ONLY_SUPPORTED: filter to entities supported on this system.
const DCGM_GEGE_FLAG_ONLY_SUPPORTED: u32 = 0x00000001;

/// DCGM_ST_OK: success return code.
const DCGM_ST_OK: i32 = 0;

/// Upper bound on CPU entity count (Grace has 2 sockets; 64 is more than enough).
const MAX_CPU_ENTITIES: usize = 64;

/// DCGM_MAX_STR_LENGTH from dcgm_structs.h.
const DCGM_MAX_STR_LENGTH: usize = 256;

/// DCGM_MAX_BLOB_LENGTH from dcgm_structs.h.
const DCGM_MAX_BLOB_LENGTH: usize = 4096;

// ── C ABI types ──────────────────────────────────────────────────────────────

/// dcgmHandle_t / dcgmGpuGrp_t / dcgmFieldGrp_t are all `uintptr_t`.
type Handle = usize;

/// dcgmGroupEntityPair_t — entity group + entity ID pair.
#[repr(C)]
#[derive(Clone, Copy)]
struct GroupEntityPair {
    entity_group_id: u32,
    entity_id: u32,
}

/// dcgmFieldValue_v2.  Layout verified against dcgm_structs.h:
///   version (u32) + entityGroupId (u32) + entityId (u32) + fieldId (u16) +
///   fieldType (u16) + status (i32) + unused (u32) + ts (i64) +
///   value union (4096 bytes) = 4128 bytes total.
#[repr(C)]
struct FieldValueV2 {
    version: u32,
    entity_group_id: u32,
    entity_id: u32,
    field_id: u16,
    field_type: u16,
    status: i32,
    _unused: u32,
    ts: i64,
    val: FieldValueUnion,
}

#[repr(C)]
union FieldValueUnion {
    i64: i64,
    dbl: f64,
    str_val: [u8; DCGM_MAX_STR_LENGTH],
    blob: [u8; DCGM_MAX_BLOB_LENGTH],
}

// ── Function pointer types (verbatim from dcgm_agent.h) ──────────────────────

type FnDcgmInit = unsafe extern "C" fn() -> i32;
type FnDcgmShutdown = unsafe extern "C" fn() -> i32;
type FnDcgmStartEmbedded = unsafe extern "C" fn(op_mode: i32, handle: *mut Handle) -> i32;
type FnDcgmStopEmbedded = unsafe extern "C" fn(handle: Handle) -> i32;
type FnDcgmGetEntityGroupEntities =
    unsafe extern "C" fn(handle: Handle, entity_group: u32, entities: *mut u32, num_entities: *mut i32, flags: u32) -> i32;
type FnDcgmGroupCreate =
    unsafe extern "C" fn(handle: Handle, group_type: i32, name: *const libc_c_char, group_id: *mut Handle) -> i32;
type FnDcgmGroupAddEntity =
    unsafe extern "C" fn(handle: Handle, group_id: Handle, entity_group_id: u32, entity_id: u32) -> i32;
type FnDcgmGroupDestroy = unsafe extern "C" fn(handle: Handle, group_id: Handle) -> i32;
type FnDcgmFieldGroupCreate = unsafe extern "C" fn(
    handle: Handle,
    num_field_ids: i32,
    field_ids: *const u16,
    name: *const libc_c_char,
    field_group_id: *mut Handle,
) -> i32;
type FnDcgmFieldGroupDestroy = unsafe extern "C" fn(handle: Handle, field_group_id: Handle) -> i32;
type FnDcgmWatchFields = unsafe extern "C" fn(
    handle: Handle,
    group_id: Handle,
    field_group_id: Handle,
    update_freq_usec: i64,
    max_keep_age_secs: f64,
    max_keep_samples: i32,
) -> i32;
type FnDcgmUpdateAllFields = unsafe extern "C" fn(handle: Handle, wait_for_update: i32) -> i32;
type FnDcgmEntitiesGetLatestValues = unsafe extern "C" fn(
    handle: Handle,
    entities: *const GroupEntityPair,
    entity_count: u32,
    fields: *const u16,
    field_count: u32,
    flags: u32,
    values: *mut FieldValueV2,
) -> i32;

// c_char is i8 on x86_64 and u8 on aarch64; use the stdlib alias so the fn
// pointer types match CString::as_ptr() on both architectures.
#[allow(non_camel_case_types)]
type libc_c_char = std::os::raw::c_char;

// ── Loaded library ────────────────────────────────────────────────────────────

/// Owns the open `libdcgm.so` handle and all resolved function pointers.
///
/// Function pointers are valid for as long as `_lib` is alive.  Because `DcgmLib`
/// is only ever accessed through `DcgmReader` (which holds it exclusively), no
/// cross-thread aliasing occurs.
struct DcgmLib {
    _lib: libloading::Library,
    init: FnDcgmInit,
    shutdown: FnDcgmShutdown,
    start_embedded: FnDcgmStartEmbedded,
    stop_embedded: FnDcgmStopEmbedded,
    get_entity_group_entities: FnDcgmGetEntityGroupEntities,
    group_create: FnDcgmGroupCreate,
    group_add_entity: FnDcgmGroupAddEntity,
    group_destroy: FnDcgmGroupDestroy,
    field_group_create: FnDcgmFieldGroupCreate,
    field_group_destroy: FnDcgmFieldGroupDestroy,
    watch_fields: FnDcgmWatchFields,
    update_all_fields: FnDcgmUpdateAllFields,
    entities_get_latest_values: FnDcgmEntitiesGetLatestValues,
}

// SAFETY: all fn pointers are derived from a loaded shared library and are
// plain C function addresses; they carry no thread-local state.
unsafe impl Send for DcgmLib {}

impl DcgmLib {
    /// Try the standard DCGM library names in version order.
    fn load() -> Result<Self, DcgmUnavailable> {
        let lib = Self::open_lib()?;
        // SAFETY: `_lib` remains alive for the lifetime of this struct, keeping
        // the resolved symbols valid.  Each symbol is transmuted from a
        // `libloading::Symbol<T>` to `T`, which is sound because Symbol<T>
        // implements `Deref<Target = T>` and fn pointers are Copy.
        unsafe {
            macro_rules! sym {
                ($name:literal, $ty:ty) => {
                    *lib.get::<$ty>($name)
                        .map_err(|e| DcgmUnavailable(format!("symbol {}: {e}", stringify!($name))))?
                };
            }
            Ok(Self {
                init: sym!(b"dcgmInit\0", FnDcgmInit),
                shutdown: sym!(b"dcgmShutdown\0", FnDcgmShutdown),
                start_embedded: sym!(b"dcgmStartEmbedded\0", FnDcgmStartEmbedded),
                stop_embedded: sym!(b"dcgmStopEmbedded\0", FnDcgmStopEmbedded),
                get_entity_group_entities: sym!(b"dcgmGetEntityGroupEntities\0", FnDcgmGetEntityGroupEntities),
                group_create: sym!(b"dcgmGroupCreate\0", FnDcgmGroupCreate),
                group_add_entity: sym!(b"dcgmGroupAddEntity\0", FnDcgmGroupAddEntity),
                group_destroy: sym!(b"dcgmGroupDestroy\0", FnDcgmGroupDestroy),
                field_group_create: sym!(b"dcgmFieldGroupCreate\0", FnDcgmFieldGroupCreate),
                field_group_destroy: sym!(b"dcgmFieldGroupDestroy\0", FnDcgmFieldGroupDestroy),
                watch_fields: sym!(b"dcgmWatchFields\0", FnDcgmWatchFields),
                update_all_fields: sym!(b"dcgmUpdateAllFields\0", FnDcgmUpdateAllFields),
                entities_get_latest_values: sym!(b"dcgmEntitiesGetLatestValues\0", FnDcgmEntitiesGetLatestValues),
                _lib: lib,
            })
        }
    }

    fn open_lib() -> Result<libloading::Library, DcgmUnavailable> {
        let names: &[&str] = &["libdcgm.so.4", "libdcgm.so.3", "libdcgm.so"];
        let mut last_err = String::new();
        for &name in names {
            match unsafe { libloading::Library::new(name) } {
                Ok(lib) => return Ok(lib),
                Err(e) => last_err = e.to_string(),
            }
        }
        Err(DcgmUnavailable(format!("libdcgm not found: {last_err}")))
    }
}

// ── Error type ────────────────────────────────────────────────────────────────

/// DCGM is not available on this host or the operation failed.
#[derive(Debug)]
pub struct DcgmUnavailable(pub String);

impl fmt::Display for DcgmUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DcgmUnavailable {}

fn check(ret: i32, context: &str) -> Result<(), DcgmUnavailable> {
    if ret == DCGM_ST_OK {
        Ok(())
    } else {
        Err(DcgmUnavailable(format!("{context}: DCGM error {ret}")))
    }
}

// ── Reader ────────────────────────────────────────────────────────────────────

/// Per-socket CPU power reading via DCGM field 1130.
///
/// `Drop` cleans up the entity group, field group, embedded handle, and library
/// in the reverse order of construction.
pub struct DcgmReader {
    lib: DcgmLib,
    handle: Handle,
    group_id: Handle,
    field_group_id: Handle,
    entities: Vec<GroupEntityPair>,
    /// CPU entity IDs in enumeration order (index == socket position).
    pub cpu_ids: Vec<u32>,
}

// SAFETY: DcgmLib is Send; all other fields are plain integers / Vecs.
unsafe impl Send for DcgmReader {}

impl DcgmReader {
    /// Load libdcgm, start embedded mode, enumerate CPU entities, register the
    /// field watch, and seed the first sample.
    pub fn new() -> Result<Self, DcgmUnavailable> {
        let lib = DcgmLib::load()?;

        // Library init — must be called once before any other DCGM function.
        check(unsafe { (lib.init)() }, "dcgmInit")?;

        // Start embedded DCGM (no external nv-hostengine required).
        let mut handle: Handle = 0;
        let ret = unsafe { (lib.start_embedded)(DCGM_OPERATION_MODE_AUTO, &mut handle) };
        if ret != DCGM_ST_OK {
            // Shutdown the library before returning the error.
            unsafe { (lib.shutdown)() };
            return Err(DcgmUnavailable(format!("dcgmStartEmbedded: error {ret}")));
        }

        match Self::setup(lib, handle) {
            Ok(reader) => Ok(reader),
            Err(e) => Err(e),
        }
    }

    fn setup(lib: DcgmLib, handle: Handle) -> Result<Self, DcgmUnavailable> {
        // Enumerate supported CPU entities.
        let mut raw_ids = vec![0u32; MAX_CPU_ENTITIES];
        let mut count: i32 = MAX_CPU_ENTITIES as i32;
        check(
            unsafe {
                (lib.get_entity_group_entities)(
                    handle,
                    DCGM_FE_CPU,
                    raw_ids.as_mut_ptr(),
                    &mut count,
                    DCGM_GEGE_FLAG_ONLY_SUPPORTED,
                )
            },
            "dcgmGetEntityGroupEntities",
        )?;

        if count <= 0 {
            return Err(DcgmUnavailable("no supported DCGM CPU entities on this host".into()));
        }
        let cpu_ids: Vec<u32> = raw_ids[..count as usize].to_vec();

        // Unique suffix prevents name collisions across concurrent processes.
        let suffix = format!("{}_{}", std::process::id(), {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        });
        let group_name = CString::new(format!("cpu_power_entities_{suffix}")).unwrap();
        let field_group_name = CString::new(format!("cpu_power_fields_{suffix}")).unwrap();

        // Create an empty entity group and add each CPU entity.
        let mut group_id: Handle = 0;
        check(
            unsafe { (lib.group_create)(handle, DCGM_GROUP_EMPTY, group_name.as_ptr(), &mut group_id) },
            "dcgmGroupCreate",
        )?;

        for &cpu_id in &cpu_ids {
            check(
                unsafe { (lib.group_add_entity)(handle, group_id, DCGM_FE_CPU, cpu_id) },
                "dcgmGroupAddEntity",
            )?;
        }

        // Create a field group containing only field 1130.
        let field_ids: [u16; 1] = [CPU_POWER_FIELD_ID];
        let mut field_group_id: Handle = 0;
        check(
            unsafe {
                (lib.field_group_create)(
                    handle,
                    1,
                    field_ids.as_ptr(),
                    field_group_name.as_ptr(),
                    &mut field_group_id,
                )
            },
            "dcgmFieldGroupCreate",
        )?;

        // Register the watch: 100 ms interval, keep 60 s of history, 600 max samples.
        check(
            unsafe { (lib.watch_fields)(handle, group_id, field_group_id, 100_000, 60.0, 600) },
            "dcgmWatchFields",
        )?;

        // Force the first watched update so the initial read returns real data
        // rather than a blank sample.  waitForUpdate=1 blocks until done.
        check(
            unsafe { (lib.update_all_fields)(handle, 1) },
            "dcgmUpdateAllFields",
        )?;

        let entities: Vec<GroupEntityPair> = cpu_ids
            .iter()
            .map(|&id| GroupEntityPair {
                entity_group_id: DCGM_FE_CPU,
                entity_id: id,
            })
            .collect();

        Ok(Self {
            lib,
            handle,
            group_id,
            field_group_id,
            entities,
            cpu_ids,
        })
    }

    /// Read the latest cached power value for each CPU entity.
    ///
    /// Returns one `(cpu_id, watts)` pair per entity.  `None` means the DCGM
    /// status for that entity was not `DCGM_ST_OK` or the value was not finite.
    pub fn read_watts(&mut self) -> Result<Vec<(u32, Option<f64>)>, DcgmUnavailable> {
        let n = self.entities.len();
        let mut values: Vec<FieldValueV2> = (0..n)
            .map(|_| FieldValueV2 {
                version: 0,
                entity_group_id: 0,
                entity_id: 0,
                field_id: 0,
                field_type: 0,
                status: 0,
                _unused: 0,
                ts: 0,
                val: FieldValueUnion { i64: 0 },
            })
            .collect();

        let field_ids: [u16; 1] = [CPU_POWER_FIELD_ID];

        check(
            unsafe {
                (self.lib.entities_get_latest_values)(
                    self.handle,
                    self.entities.as_ptr(),
                    n as u32,
                    field_ids.as_ptr(),
                    1,
                    0, // flags=0: use cached data — DCGM_FV_FLAG_LIVE_DATA breaks watched fields
                    values.as_mut_ptr(),
                )
            },
            "dcgmEntitiesGetLatestValues",
        )?;

        let result: Vec<(u32, Option<f64>)> = values
            .iter()
            .map(|v| {
                let watts = if v.status == DCGM_ST_OK {
                    // SAFETY: field 1130 is a double field; union variant is valid.
                    let w = unsafe { v.val.dbl };
                    if w.is_finite() && w > 0.0 {
                        Some(w)
                    } else {
                        None
                    }
                } else {
                    None
                };
                (v.entity_id, watts)
            })
            .collect();

        Ok(result)
    }
}

impl Drop for DcgmReader {
    fn drop(&mut self) {
        // Best-effort cleanup in reverse-construction order.
        // Errors are swallowed — a failed unwatch must not panic at shutdown.
        unsafe {
            (self.lib.watch_fields)(
                self.handle,
                self.group_id,
                self.field_group_id,
                0,   // updateFreq=0 means unwatch
                0.0,
                0,
            );
            (self.lib.field_group_destroy)(self.handle, self.field_group_id);
            (self.lib.group_destroy)(self.handle, self.group_id);
            (self.lib.stop_embedded)(self.handle);
            (self.lib.shutdown)();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_value_v2_size() {
        // Verify our struct matches the C layout: 32 bytes of header + 4096 byte union.
        assert_eq!(std::mem::size_of::<FieldValueV2>(), 4128);
        assert_eq!(std::mem::align_of::<FieldValueV2>(), 8);
    }

    #[test]
    fn group_entity_pair_size() {
        assert_eq!(std::mem::size_of::<GroupEntityPair>(), 8);
    }

    #[test]
    fn dcgm_unavailable_on_missing_lib() {
        // On hosts without libdcgm.so this must be a soft error, not a panic.
        match DcgmReader::new() {
            Ok(_) => {}
            Err(e) => {
                // Must not be empty — callers log this message.
                assert!(!e.0.is_empty(), "DcgmUnavailable must carry a message");
            }
        }
    }
}
