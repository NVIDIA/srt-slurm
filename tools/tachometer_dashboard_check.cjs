// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Capture-independent regression using local headless Chrome. NODE_PATH may point at a separate playwright-core install.
// node tools/tachometer_dashboard_check.cjs dashboard.html screenshot-directory
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright-core');

async function main() {
  const [input, outputDir] = process.argv.slice(2);
  if (!input || !outputDir) throw new Error('Usage: node tachometer_dashboard_check.cjs HTML OUTPUT_DIR');
  fs.mkdirSync(outputDir, {recursive: true});
  const browser = await chromium.launch({executablePath: '/usr/bin/google-chrome', headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const context = await browser.newContext({viewport: {width: 1600, height: 1100}, reducedMotion: 'reduce'});
  await context.setOffline(true);
  const page = await context.newPage();
  const errors = [], network = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('request', r => {if (/^https?:/.test(r.url())) network.push(r.url());});
  try {
    await page.goto(pathToFileURL(path.resolve(input)).href, {waitUntil: 'load', timeout: 60000});
    await page.waitForFunction(() => window.__tachometerDashboard?.ready, null, {timeout: 30000});
    // Corrupt but valid JSON must fall back to usable controls, not prevent initialization.
    for (const saved of [null, true, {filters:"invalid",collapsed:[],views:42,selections:{m0000:{invalid:true}}}]) {
      await page.evaluate(saved => {
        for (const key of Object.keys(localStorage).filter(k => k.startsWith('tachometer-dashboard-'))) {
          localStorage.setItem(key, JSON.stringify(saved));
        }
      }, saved);
      await page.reload({waitUntil:'load', timeout:60000});
      await page.waitForFunction(() => window.__tachometerDashboard?.ready);
      assert.deepEqual(await page.evaluate(() => {
        const state=window.__tachometerDashboard.state;
        return [state.filters,state.views,state.selections];
      }), [{},{},{}]);
    }
    // Reopening a dashboard with the old default must not silently hide families.
    await page.evaluate(() => {
      for (const key of Object.keys(localStorage).filter(k => k.startsWith('tachometer-dashboard-'))) {
        localStorage.setItem(key, JSON.stringify({all:false,component:'Frontend'}));
      }
    });
    await page.reload({waitUntil:'load', timeout:60000});
    await page.waitForFunction(() => window.__tachometerDashboard?.ready);
    assert.equal(await page.locator('#featured-only').isChecked(), false);
    const report = await page.evaluate(() => {
      const api = window.__tachometerDashboard;
      return {title: document.title, rows: api.catalog.row_count, families: api.catalog.metrics.length,
        series: api.catalog.metrics.reduce((n, m) => n + m.series_count, 0), tabs: [...document.querySelectorAll('#tabs button')].map(n => n.textContent)};
    });
    assert.equal(report.tabs.length, 5);
    report.excludedSourceFiles=await page.evaluate(()=>(window.__tachometerDashboard.catalog.excluded_source_files || []).length);
    report.hasFinalSource=await page.evaluate(()=>(window.__tachometerDashboard.catalog.source_files || []).some(source=>String(typeof source==='string'?source:source.path || '').split(/[\\/]/).pop()==='final.parquet'));
    assert.equal(await page.locator('#capture-warning').count(),report.excludedSourceFiles || !report.hasFinalSource?1:0);
    if(report.excludedSourceFiles || !report.hasFinalSource){
      if(!report.hasFinalSource) assert.match(await page.locator('#capture-warning').textContent(),/may be incomplete: no final.parquet/);
      await page.locator('#capture-warning').click();
      const details=await page.locator('#inspector-body').textContent();
      assert.match(details,/excluded_source_files/);
      if(!report.hasFinalSource) assert.match(details,/No final.parquet was selected/);
      await page.locator('#close-inspector').click();
    }
    async function panelCounts() {
      return page.evaluate(() => {
        const api = window.__tachometerDashboard;
        const tab = document.querySelector('#tabs button[aria-selected="true"]');
        const panels = document.querySelectorAll('.metric-panel').length;
        const rows = [...document.querySelectorAll('.metric-row')];
        const rowSum = rows.reduce((n,row) => n + Number(row.dataset.count), 0);
        const expected = api.catalog.metrics.filter(m => m.component === api.state.component &&
          (!api.state.featuredOnly || m.featured) && (!api.state.search || [m.name,m.title,m.group,m.component,...(m.aliases||[])].join(" ").toLowerCase().includes(api.state.search.toLowerCase()))).length;
        if (panels !== rowSum || panels !== Number(tab.dataset.visible) || panels !== Number(document.querySelector('#panel-count').dataset.visible)) {
          throw new Error('Tab/row/panel mismatch: ' + JSON.stringify({panels,rowSum,tab:tab.dataset.visible}));
        }
        for (const row of rows) if (Number(row.dataset.count) !== row.querySelectorAll('.metric-panel').length) throw new Error('Subgroup count mismatch');
        return {component:api.state.component,panels,rowSum,tabVisible:Number(tab.dataset.visible),tabTotal:Number(tab.dataset.total),expected};
      });
    }
    report.panelCounts = [];
    for (const component of ['Frontend', 'Router', 'Workers', 'GPU', 'Host']) {
      await page.evaluate(name => window.__tachometerDashboard.setTab(name), component);
      await page.waitForFunction(() => ![...document.querySelectorAll('.metric-row[open] .metric-panel')].filter(n=>n.getBoundingClientRect().top < innerHeight+200).some(n=>n.querySelector('.chart-message')?.textContent==='Loading metric…'), null, {timeout: 30000});
      const counts = await panelCounts();
      assert.equal(counts.panels, counts.tabTotal, 'All captured families must have panels by default');
      assert.equal(counts.panels, counts.expected);
      report.panelCounts.push(counts);
      await page.screenshot({path: path.join(outputDir, component.toLowerCase() + '.png')});
    }
    assert.equal(report.panelCounts.reduce((n,c) => n + c.panels,0), report.families);
    await page.evaluate(() => window.__tachometerDashboard.setTab('Frontend'));
    await page.locator('#featured-only').check();
    report.featuredCounts = await panelCounts();
    assert.equal(report.featuredCounts.panels, report.featuredCounts.expected);
    if(report.featuredCounts.panels < report.featuredCounts.tabTotal) assert.match(await page.locator('#tabs button[aria-selected="true"] .tab-count').textContent(), /\//);
    await page.screenshot({path:path.join(outputDir,'featured-only.png')});
    await page.locator('#featured-only').uncheck();
    // Every compressed payload must load and retain the advertised full series identities.
    report.loadedFamilies = await page.evaluate(async () => {
      const api = window.__tachometerDashboard;
      for (const metric of api.catalog.metrics) {
        const payload = await api.loadMetric(metric);
        if (payload.series.length !== metric.series_count) throw new Error('Series-count mismatch: ' + metric.name);
        const ids = new Set();
        for (const series of payload.series) {
          const id = JSON.stringify([series.endpoint, Object.entries(series.metadata).sort(), Object.entries(series.labels).sort()]);
          if (ids.has(id)) throw new Error('Duplicate full identity: ' + metric.name);
          ids.add(id);
        }
      }
      return api.catalog.metrics.length;
    });
    assert.equal(report.loadedFamilies, report.families);
    // Any captured histogram alias must find exactly its parent family.
    const aliasMetric = await page.evaluate(() => window.__tachometerDashboard.catalog.metrics.find(m => m.kind === 'histogram' && m.aliases?.length));
    const inspectedMetric = aliasMetric || await page.evaluate(() => window.__tachometerDashboard.catalog.metrics[0]);
    if (inspectedMetric) {
      await page.evaluate(name => window.__tachometerDashboard.setTab(name), inspectedMetric.component);
      await page.locator('#metric-search').fill(aliasMetric ? aliasMetric.aliases[0] : inspectedMetric.name);
      await page.waitForFunction(id => document.querySelector('#metric-'+id), inspectedMetric.id);
      await page.waitForTimeout(200);
      const searchCounts = await panelCounts();
      assert.equal(searchCounts.panels, searchCounts.expected);
      const target = page.locator('#metric-'+inspectedMetric.id);
      await target.getByRole('button', {name:'Inspect',exact:true}).click();
      await page.waitForSelector('#inspector[open]');
      assert.match(await page.locator('#inspector-body').textContent(), /label|endpoint/i);
      await page.screenshot({path:path.join(outputDir,'inspector.png')});
      await page.locator('#close-inspector').click();
      await page.locator('#metric-search').fill('');
      await page.waitForTimeout(200);
    }
    // Time selection survives changing tabs and updates every visible chart scale.
    const end = await page.evaluate(() => window.__tachometerDashboard.catalog.duration_s);
    await page.locator('#time-from').fill(String(Math.min(100, end / 4)));
    await page.locator('#time-to').fill(String(Math.min(1000, end / 2)));
    await page.locator('#apply-time').click();
    const selectedTime = await page.evaluate(() => [window.__tachometerDashboard.state.from, window.__tachometerDashboard.state.to]);
    await page.evaluate(() => window.__tachometerDashboard.setTab('GPU'));
    assert.deepEqual(await page.evaluate(() => [window.__tachometerDashboard.state.from, window.__tachometerDashboard.state.to]), selectedTime);
    await page.waitForTimeout(300);
    await page.screenshot({path: path.join(outputDir, 'gpu-zoomed.png')});
    await page.locator('#reset-time').click();
    // Pick an actual scalar source, then filter by its recorded endpoint and one label.
    const chosen = await page.evaluate(async () => {
      const api=window.__tachometerDashboard;
      const m=api.catalog.metrics.find(m=>m.kind==='scalar' && m.series_count>1) || api.catalog.metrics.find(m=>m.kind==='scalar');
      if(!m) return null;
      const payload=await api.loadMetric(m), series=payload.series[0];
      const labels={...series.metadata,...series.labels,endpoint:series.endpoint};
      const key=['gpu','worker_role','groupname','hostname'].find(k=>labels[k]!==undefined);
      return {metric:m,key,value:key?String(labels[key]):null,endpoint:series.endpoint};
    });
    if(chosen) {
      await page.evaluate(c=>window.__tachometerDashboard.setTab(c),chosen.metric.component);
      await page.getByLabel('Filter endpoint',{exact:true}).selectOption(chosen.endpoint);
      if(chosen.key && await page.getByLabel('Filter '+chosen.key,{exact:true}).count()) await page.getByLabel('Filter '+chosen.key,{exact:true}).selectOption(chosen.value);
      const filtered=await page.evaluate(async ({metric})=> {
        const api=window.__tachometerDashboard;
        return api.buildSeries(metric,await api.loadMetric(metric),metric.counter?'rate':'mean').series.map(s=>({endpoint:s.endpoint,labels:{...s.metadata,...s.labels}}));
      },chosen);
      assert(filtered.length>0);
      assert(filtered.every(s=>s.endpoint===chosen.endpoint));
      if(chosen.key) assert(filtered.every(s=>String(s.labels[chosen.key])===chosen.value));
      report.filteredSeries=filtered.length;
      const row=page.locator('#metric-'+chosen.metric.id).locator('..').locator('..');
      await row.evaluate(n=>n.open=true);
      await page.locator('#metric-'+chosen.metric.id).scrollIntoViewIfNeeded();
      await page.waitForTimeout(500);
      const legend=page.locator('#metric-'+chosen.metric.id+' .legend-button').first();
      if(await legend.count()) { await legend.click(); assert.equal(await legend.getAttribute('aria-pressed'),'false'); }
      await page.screenshot({path:path.join(outputDir,'series-filtered.png')});
      await page.locator('#clear-filters').click();
    }
    // Bucket arithmetic is checked independently of rendered pixels.
    const histogram = await page.evaluate(() => {
      const api = window.__tachometerDashboard;
      const columns = ['bin','mean','min','max','last','delta','observed_s','samples','resets','gaps'];
      const counts = [[20,20,0],[40,80,0],[100,100,0]];
      const payload = {columns,series:['1','5','+Inf'].map((le,i) => ({id:'s'+i,endpoint:'frontend0',metadata:{},labels:{le},points:counts[i].map((count,bin)=>[bin,0,0,0,0,count,10,2,0,0])}))};
      const r = api.buildSeries({kind:'histogram'},payload,'p50');
      payload.series[0].points[1][6] = 5;
      const unequal = api.buildSeries({kind:'histogram'},payload,'p50');
      return {points:r.series[0].points,overflow:r.overflow,unequal:unequal.series[0].points[1]};
    });
    assert.deepEqual(histogram.points, [[0,null],[1,3],[2,null]]);
    assert.equal(histogram.overflow, 1);
    assert.deepEqual(histogram.unequal, [1,null], 'Unequal histogram intervals must not produce a percentile');
    // Full-catalog explorer and all-metric rows remain available beyond featured panels.
    await page.locator('#explore-button').click();
    await page.waitForSelector('#explorer:not([hidden])');
    assert.equal(await page.locator('.explorer-item').count(),report.families);
    await page.screenshot({path: path.join(outputDir, 'explorer.png')});
    await page.locator('#close-explorer').click();
    await page.locator('#featured-only').uncheck();
    await page.evaluate(() => window.__tachometerDashboard.setTab('GPU'));
    const gpuExpected = await page.evaluate(() => window.__tachometerDashboard.catalog.metrics.filter(m => m.component === 'GPU').length);
    assert.equal(await page.locator('.metric-panel').count(), gpuExpected);
    await page.locator('#collapse-all').click();
    const collapsedCounts = await panelCounts();
    assert.equal(collapsedCounts.panels,gpuExpected);
    assert.equal(await page.locator('.metric-row[open]').count(),0);
    await page.locator('#expand-all').click();
    // Small-screen rendering should keep controls usable without page-wide overflow.
    await page.setViewportSize({width: 760, height: 1000});
    await page.screenshot({path: path.join(outputDir, 'narrow.png')});
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 2);
    assert.equal(overflow, false, 'Dashboard overflows the viewport horizontally');
    assert.deepEqual(errors, [], 'Browser errors');
    assert.deepEqual(network, [], 'Offline dashboard attempted external requests');
    report.errors = errors; report.externalRequests = network; report.passed = true;
    fs.writeFileSync(path.join(outputDir, 'browser-report.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2));
  } finally { await browser.close(); }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
