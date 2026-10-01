/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */
(() => {
  "use strict";
  const overlaps = (c, from, to) => c.bounds[0] <= to && c.bounds[3] >= from;
  function create(data) {
    const cache = new Map();
    const budget = data.delivery?.cache_bytes ?? 32 * 1024 * 1024;
    let bytes = 0;
    async function load(chunk, signal) {
      signal?.throwIfAborted();
      if (cache.has(chunk.url)) {
        const hit = cache.get(chunk.url);
        cache.delete(chunk.url); cache.set(chunk.url, hit);
        return hit.rows;
      }
      if (location.protocol === "file:")
        throw Error("This report loads detail files. Serve the complete directory over HTTP, or rebuild with --single-file.");
      const response = await fetch(new URL(chunk.url, location.href), {signal});
      if (!response.ok) throw Error(`Could not load report detail (${response.status}). Copy the complete report directory.`);
      let raw = new Uint8Array(await response.arrayBuffer());
      // Static hosts may already have decoded Content-Encoding: gzip.
      if (raw[0] === 31 && raw[1] === 139)
        raw = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
      signal?.throwIfAborted();
      if (raw.byteLength !== chunk.decoded_bytes) throw Error("Report detail size mismatch; reload the report generation.");
      if (crypto.subtle) {
        const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", raw))].map(b => b.toString(16).padStart(2, "0")).join("");
        if (hash !== chunk.decoded_sha256) throw Error("Report detail hash mismatch; reload the report generation.");
      }
      const rows = JSON.parse(new TextDecoder().decode(raw));
      if (rows.length !== chunk.count) throw Error("Report detail row count mismatch.");
      signal?.throwIfAborted();
      if (chunk.decoded_bytes <= budget) {
        // A concurrent reader may already have installed this immutable shard.
        if (cache.has(chunk.url)) bytes -= cache.get(chunk.url).bytes;
        cache.delete(chunk.url);
        while (bytes + chunk.decoded_bytes > budget && cache.size) {
          const oldest = cache.keys().next().value;
          bytes -= cache.get(oldest).bytes; cache.delete(oldest);
        }
        cache.set(chunk.url, {rows, bytes: chunk.decoded_bytes}); bytes += chunk.decoded_bytes;
      }
      return rows;
    }
    async function events(profile, {from, to, name = "", offset = 0, limit = 100, signal} = {}) {
      const items = [];
      let total = 0;
      const search = name.toLowerCase();
      for (const chunk of profile.event_chunks) {
        signal?.throwIfAborted();
        if (!overlaps(chunk, from, to)) continue;
        const complete = !search && chunk.bounds[0] >= from && chunk.bounds[3] <= to;
        // Exact counts from wholly selected shards need no download. Decode only
        // boundary/name-filtered shards and the page the caller requested.
        if (complete && (total + chunk.count <= offset || items.length >= limit)) {
          total += chunk.count; continue;
        }
        const rows = await load(chunk, signal);
        for (const e of rows) {
          if (e[0] > to || e[1] < from || (search && !profile.names[e[2]].toLowerCase().includes(search))) continue;
          if (total >= offset && items.length < limit) items.push(e);
          total++;
        }
      }
      return {items, total, offset, limit, range: [from, to], partial: Boolean(profile.truncated)};
    }
    async function metric(series, from, to, signal) {
      const selected = series.chunks.filter(c => overlaps(c, from, to));
      // Settings carry the complete last timestamp group into a later window.
      // Other series retain one neighboring shard per side for chart continuity.
      const before = series.chunks.filter(c => c.bounds[3] < from);
      const priorTime = Math.max(-Infinity, ...before.map(c => c.bounds[3]));
      const prior = series.chunks.filter(c => c.bounds[0] < from && c.bounds[3] >= priorTime);
      const after = series.chunks.find(c => c.bounds[0] > to);
      const wanted = new Set([...selected, ...prior, ...(after ? [after] : [])]);
      const points = [];
      for (const chunk of series.chunks) {
        if (wanted.has(chunk)) points.push(...await load(chunk, signal));
      }
      return points;
    }
    async function cpu(profile, from, to, signal) {
      const cpu = profile.cpu;
      let total = 0;
      const counts = new Map();
      for (const chunk of cpu?.chunks || []) {
        if (!overlaps(chunk, from, to)) continue;
        for (const s of await load(chunk, signal)) {
          if (s[0] < from || s[0] > to) continue;
          total++;
          for (const id of new Set(cpu.stacks[s[2]])) counts.set(id, (counts.get(id) || 0) + 1);
        }
      }
      return {counts, total};
    }
    return Object.freeze({load, events, metric, cpu,
      estimate: (profile, from, to) => profile.event_chunks.filter(c => overlaps(c, from, to)).reduce((n, c) => n + c.count, 0),
      status: () => ({cachedChunks: cache.size, decodedBytes: bytes, maxDecodedBytes: budget}),
    });
  }
  window.DSightDetailData = Object.freeze({create});
})();
