// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Headless render check for a generated perf dashboard.
//
// `node --check` proves the page's JS parses. It does not prove the page BUILDS:
// a page whose script throws on line one is syntactically perfect and completely
// blank. This executes the real page in a DOM and asserts panels were actually
// created, which is the difference between "looks fine" and "renders".
//
// Deliberately not wired into CI: it needs npm deps (jsdom, d3) that a Python
// repo should not carry. Run it by hand when changing the renderer's JS.
//
//   npm install jsdom d3@7
//   node tools/dashboard_render_check.js <perf_dashboard.html>
//
// Exit 0 = rendered, 1 = threw or produced nothing.

// node --check only proves the JS parses; this proves it runs to completion and
// produced panels, which is the difference between "looks fine" and "renders".
const fs = require('fs');
const { JSDOM } = require('jsdom');
const d3 = require('d3');

const file = process.argv[2];
const html = fs.readFileSync(file, 'utf8');
const dom = new JSDOM(html, { runScripts: 'outside-only', pretendToBeVisual: true });
const { window } = dom;
// d3's Node build resolves `document` from the GLOBAL scope, not from a window
// object handed to it, so the jsdom document has to be published globally before
// any d3.select runs.
global.window = window;
global.document = window.document;
global.navigator = window.navigator;
global.location = window.location;
window.d3 = d3;
global.d3 = d3;
const errors = [];
window.addEventListener('error', e => errors.push(String(e.error || e.message)));

const body = html.match(/<script>([\s\S]*?)<\/script>/g)
  .map(b => b.replace(/^<script>/, '').replace(/<\/script>$/, ''))
  .sort((a, b) => b.length - a.length)[0];

try { new Function('d3','document','window','location', body)(d3, window.document, window, window.location); } catch (e) { errors.push(e.stack || String(e)); }

const doc = window.document;
const tabs = [...doc.querySelectorAll('.tab')].map(t => t.textContent);
const panels = doc.querySelectorAll('.panel').length;
const svgs = doc.querySelectorAll('svg').length;
const paths = doc.querySelectorAll('path').length;
const rows = doc.querySelectorAll('.tbl tbody tr').length;

console.log('tabs   :', tabs.join(' | '));
console.log('panels :', panels);
console.log('svgs   :', svgs, ' paths:', paths);
console.log('session rows:', rows);
if (errors.length) { console.log('ERRORS :'); errors.forEach(e => console.log('  ' + e.split('\n')[0])); }

const ok = errors.length === 0 && panels > 0 && svgs > 0;
console.log(ok ? 'RENDER OK' : 'RENDER FAILED');
process.exit(ok ? 0 : 1);
