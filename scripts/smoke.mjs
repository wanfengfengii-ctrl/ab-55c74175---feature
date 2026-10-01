#!/usr/bin/env node
/**
 * HTTP smoke check against a running API instance. Posts the canonical
 * cross-week + missing-packet sample and asserts the recovered interpretation.
 *
 * Usage: node scripts/smoke.mjs [baseUrl]
 * Exit code 0 on success, 1 on any failure.
 */

const baseUrl = process.argv[2] ?? process.env.API_BASE_URL ?? 'http://127.0.0.1:3000';

const sample = {
  modulus: 10,
  countLower: 0,
  countUpper: 120,
  minInterval: 9,
  maxInterval: 11,
  packets: [
    { id: 'G', remainder: 1, timeLower: 307, timeUpper: 313 },
    { id: 'A', remainder: 8, timeLower: 77, timeUpper: 83 },
    { id: 'F', remainder: 0, timeLower: 297, timeUpper: 303 },
    { id: 'C', remainder: 2, timeLower: 117, timeUpper: 123 },
    { id: 'B', remainder: 9, timeLower: 87, timeUpper: 93 },
    { id: 'E', remainder: 2, timeLower: 217, timeUpper: 223 },
    { id: 'D', remainder: 1, timeLower: 207, timeUpper: 213 },
  ],
};

const expectedOrder = ['A', 'B', 'C', 'D', 'E', 'F', 'G'];
const expectedCounts = [8, 9, 12, 21, 22, 30, 31];
const expectedMissing = [
  [10, 11],
  [13, 20],
  [23, 29],
];

function fail(message) {
  console.error(`SMOKE FAILED: ${message}`);
  process.exit(1);
}

async function main() {
  // 1. Health endpoint.
  const healthRes = await fetch(`${baseUrl}/health`);
  if (!healthRes.ok) fail(`GET /health returned ${healthRes.status}`);
  const health = await healthRes.json();
  if (health.status !== 'ok') fail(`health payload not ok: ${JSON.stringify(health)}`);

  // 2. Recovery on the cross-week sample.
  const res = await fetch(`${baseUrl}/api/v1/recover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(sample),
  });
  if (res.status !== 200) {
    fail(`POST /api/v1/recover returned ${res.status}: ${await res.text()}`);
  }
  const body = await res.json();
  if (body.status !== 'ok') fail(`response status not ok: ${JSON.stringify(body)}`);

  const { data } = body;
  if (JSON.stringify(data.order) !== JSON.stringify(expectedOrder)) {
    fail(`wrong order: got ${JSON.stringify(data.order)}`);
  }
  const counts = data.assignments.map((a) => a.absoluteCount);
  if (JSON.stringify(counts) !== JSON.stringify(expectedCounts)) {
    fail(`wrong absolute counts: got ${JSON.stringify(counts)}`);
  }
  for (let k = 1; k < data.assignments.length; k++) {
    const prev = data.assignments[k - 1];
    const cur = data.assignments[k];
    if (cur.absoluteCount <= prev.absoluteCount) fail('counts not strictly increasing');
    if (cur.time <= prev.time) fail('timestamps not strictly increasing');
    if (((cur.absoluteCount % 10) + 10) % 10 !== cur.remainder) fail('count/remainder mismatch');
    if (cur.time < cur.timeInterval.lower || cur.time > cur.timeInterval.upper) {
      fail('selected time outside packet closed interval');
    }
  }
  const segments = data.missingSegments.map((s) => [s.fromCount, s.toCount]);
  if (JSON.stringify(segments) !== JSON.stringify(expectedMissing)) {
    fail(`wrong missing segments: got ${JSON.stringify(segments)}`);
  }
  if (data.missingCountTotal !== 17) fail(`wrong missing total: ${data.missingCountTotal}`);
  if (data.adjacency.length !== 6) fail('expected 6 adjacency evidence entries');
  for (const ev of data.adjacency) {
    if (!ev.satisfied) fail(`unsatisfied adjacency evidence: ${JSON.stringify(ev)}`);
    if (ev.timeGap < ev.allowedTimeGap.min || ev.timeGap > ev.allowedTimeGap.max) {
      fail(`time gap ${ev.timeGap} outside [${ev.allowedTimeGap.min}, ${ev.allowedTimeGap.max}]`);
    }
  }

  // 3. Infeasible request must surface the stable business error code.
  const bad = await fetch(`${baseUrl}/api/v1/recover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...sample, countUpper: 3 }),
  });
  if (bad.status !== 400 && bad.status !== 422) {
    fail(`infeasible request returned HTTP ${bad.status}`);
  }
  const badBody = await bad.json();
  if (badBody.status !== 'error' || !badBody.error.code) fail('error body missing stable code');

  // 4. Cumulative-jitter budget (nominalInterval + totalJitterBudget). The
  // sample timestamps are exact 10-ticks-per-count multiples, so nominal
  // interval 10 and a zero budget must be exactly exhausted.
  const resJ = await fetch(`${baseUrl}/api/v1/recover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...sample, nominalInterval: 10, totalJitterBudget: 0 }),
  });
  if (resJ.status !== 200) {
    fail(`budgeted request returned ${resJ.status}: ${await resJ.text()}`);
  }
  const bodyJ = await resJ.json();
  if (bodyJ.status !== 'ok') fail(`budgeted response not ok: ${JSON.stringify(bodyJ)}`);
  const d = bodyJ.data;
  if (JSON.stringify(d.jitterBudget) !== JSON.stringify({
    nominalInterval: 10, budget: 0, used: 0, remaining: 0, exhausted: true,
  })) {
    fail(`wrong jitterBudget summary: ${JSON.stringify(d.jitterBudget)}`);
  }
  for (let i = 0; i < d.adjacency.length; i++) {
    const ev = d.adjacency[i];
    if (ev.nominalTimeGap !== ev.countGap * 10) fail('wrong nominalTimeGap');
    if (ev.jitter !== Math.abs(ev.timeGap - ev.nominalTimeGap)) fail('wrong per-edge jitter');
    if (ev.cumulativeJitter !== ev.jitter + (i === 0 ? 0 : d.adjacency[i - 1].cumulativeJitter)) {
      fail('cumulativeJitter is not a running total');
    }
  }

  // 5. A budget the same order cannot meet must return 422 with the first
  // blocking evidence (used, minimumAdditional, budget).
  const resTight = await fetch(`${baseUrl}/api/v1/recover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...sample, nominalInterval: 11, totalJitterBudget: 0 }),
  });
  if (resTight.status !== 422) {
    fail(`over-tight budget expected 422, got ${resTight.status}`);
  }
  const bodyTight = await resTight.json();
  if (bodyTight.error?.code !== 'NO_CONSISTENT_INTERPRETATION') {
    fail(`over-tight budget wrong error code: ${JSON.stringify(bodyTight)}`);
  }
  const jd = bodyTight.error?.evidence?.detail?.jitter;
  if (!jd || typeof jd.used !== 'number' || typeof jd.minimumAdditional !== 'number' || jd.budget !== 0) {
    fail(`jitter blocker evidence missing used/minimumAdditional/budget: ${JSON.stringify(bodyTight)}`);
  }

  // 6. Out-of-range nominalInterval is a request-validation error (400).
  const resInvalid = await fetch(`${baseUrl}/api/v1/recover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...sample, nominalInterval: 99, totalJitterBudget: 0 }),
  });
  if (resInvalid.status !== 400) {
    fail(`out-of-range nominal expected 400, got ${resInvalid.status}`);
  }
  const bodyInvalid = await resInvalid.json();
  if (bodyInvalid.error?.code !== 'INVALID_REQUEST') fail('expected INVALID_REQUEST code');

  console.log('SMOKE PASSED');
  console.log(`  order     : ${data.order.join(' -> ')}`);
  console.log(`  counts    : ${counts.join(', ')}`);
  console.log(`  missing   : ${data.missingCountTotal} packets in ${segments.length} segment(s)`);
  console.log(`  adjacency : all ${data.adjacency.length} constraints satisfied`);
}

main().catch((err) => fail(err.stack ?? String(err)));
