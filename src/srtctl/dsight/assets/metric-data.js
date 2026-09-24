/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */
(() => {
  "use strict";
  const clone = value => JSON.parse(JSON.stringify(value));

  function create(data, {maxCachedPoints = 500000} = {}) {
    const byName = new Map();
    for (const series of data.metrics) {
      if (!byName.has(series.name)) byName.set(series.name, []);
      byName.get(series.name).push(series);
    }
    const families = data.metric_catalog || [...byName].map(([name, series]) => ({
      name, title: series[0].label || name, unit: series[0].unit || "",
      component: "Recorded metrics", group: "Metrics", order: 0,
      series_count: series.length, samples: series.reduce((n, s) => n + (s.points?.length || 0), 0),
      value_kind: "stored", description: "Recorded values.",
    }));
    const familyByName = new Map(families.map(family => [family.name, family]));
    const cache = new Map(), pending = new Map();
    let cachedPoints = 0;
    // Serial decompression also bounds temporary JSON/base64 memory during broad queries.
    let queue = Promise.resolve();
    const metadata = series => {
      const {points, ...rest} = series;
      return rest;
    };
    function matching({name, worker, host, gpu, rank} = {}) {
      const candidates = name === undefined ? data.metrics : byName.get(name) || [];
      return candidates.filter(s =>
        (worker === undefined || s.worker === worker) &&
        (host === undefined || s.host === host) &&
        (gpu === undefined || s.gpu === String(gpu)) &&
        (rank === undefined || String(s.rank) === String(rank)));
    }
    async function decode(name) {
      const source = byName.get(name) || [];
      const payloadId = data.metric_payloads?.[name];
      if (!payloadId) {
        if (data.metric_payloads && source.some(s => !Array.isArray(s.points)))
          throw Error(`Missing embedded metric data for ${name}`);
        return source;
      }
      const payload = document.getElementById(payloadId);
      if (!payload) throw Error(`Missing embedded metric payload: ${payloadId}`);
      const binary = atob(payload.textContent.trim());
      const bytes = Uint8Array.from(binary, c => c.charCodeAt(0));
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
      const points = JSON.parse(await new Response(stream).text());
      return source.map(series => {
        if (!Array.isArray(points[String(series.id)]))
          throw Error(`Missing points for metric series ${series.id}`);
        return {...series, points: points[String(series.id)]};
      });
    }
    function loadFamily(name) {
      if (!familyByName.has(name)) return Promise.reject(Error(`Unknown metric family: ${name}`));
      if (cache.has(name)) {
        const entry = cache.get(name);
        cache.delete(name); cache.set(name, entry);
        return Promise.resolve(entry.series);
      }
      if (pending.has(name)) return pending.get(name);
      const request = queue.then(() => decode(name)).then(series => {
        const cost = series.reduce((n, s) => n + s.points.length, 0);
        // An oversized family can be displayed, but is never retained by the cache.
        if (cost <= maxCachedPoints) {
          while (cachedPoints + cost > maxCachedPoints && cache.size) {
            const oldest = cache.keys().next().value;
            cachedPoints -= cache.get(oldest).cost;
            cache.delete(oldest);
          }
          cache.set(name, {series, cost}); cachedPoints += cost;
        }
        return series;
      });
      pending.set(name, request);
      queue = request.catch(() => {});
      request.then(() => pending.delete(name), () => pending.delete(name));
      return request;
    }
    return Object.freeze({
      loadFamily,
      listFamilies: () => clone(families),
      listSeries: filters => matching(filters).map(series => clone(metadata(series))),
      status: () => ({
        cachedFamilies: [...cache.keys()], cachedPoints, maxCachedPoints,
        pendingFamilies: [...pending.keys()],
      }),
    });
  }
  window.DSightMetricData = Object.freeze({create});
})();
