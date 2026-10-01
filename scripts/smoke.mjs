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

  // 4. Cumulative jitter budget.
  // 4a. Backward compatibility: absent parameters add no fields.
  if (data.jitter !== undefined) fail('jitter summary must be absent without budget parameters');
  for (const ev of data.adjacency) {
    if (ev.jitter !== undefined || ev.nominalTimeGap !== undefined || ev.cumulativeJitter !== undefined) {
      fail('per-edge jitter fields must be absent without budget parameters');
    }
  }

  // 4b. The true timeline is exactly nominal (interval 10), so budget 0 must
  // keep the same interpretation and report a fully exhausted zero budget.
  const zeroRes = await fetch(`${baseUrl}/api/v1/recover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...sample, nominalInterval: 10, totalJitterBudget: 0 }),
  });
  if (zeroRes.status !== 200) fail(`budget-0 recovery returned ${zeroRes.status}: ${await zeroRes.text()}`);
  const zeroBody = await zeroRes.json();
  if (zeroBody.status !== 'ok') fail(`budget-0 recovery not ok: ${JSON.stringify(zeroBody)}`);
  if (JSON.stringify(zeroBody.data.order) !== JSON.stringify(expectedOrder)) {
    fail(`budget-0 changed the order: ${JSON.stringify(zeroBody.data.order)}`);
  }
  if (JSON.stringify(zeroBody.data.assignments.map((a) => a.absoluteCount)) !== JSON.stringify(expectedCounts)) {
    fail('budget-0 changed absolute counts');
  }
  const zj = zeroBody.data.jitter;
  if (!zj || zj.nominalInterval !== 10 || zj.budget !== 0 || zj.used !== 0 || zj.remaining !== 0 || zj.exhausted !== true) {
    fail(`budget-0 jitter summary wrong: ${JSON.stringify(zj)}`);
  }
  let cumulative = 0;
  for (const ev of zeroBody.data.adjacency) {
    if (ev.nominalTimeGap !== ev.countGap * 10) fail('nominalTimeGap mismatch');
    cumulative += ev.jitter;
    if (ev.cumulativeJitter !== cumulative) fail('cumulativeJitter accounting mismatch');
  }
  if (cumulative !== 0) fail('budget-0 sample must have zero total jitter');

  // 4c. Budget exactly exhausted is accepted (budget 0 above). A loose budget
  // reports the same solution with a non-exhausted summary.
  const looseRes = await fetch(`${baseUrl}/api/v1/recover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...sample, nominalInterval: 10, totalJitterBudget: 42 }),
  });
  if (looseRes.status !== 200) fail(`loose-budget recovery returned ${looseRes.status}`);
  const looseBody = await looseRes.json();
  if (looseBody.data.jitter.used !== 0 || looseBody.data.jitter.remaining !== 42 || looseBody.data.jitter.exhausted !== false) {
    fail(`loose-budget summary wrong: ${JSON.stringify(looseBody.data.jitter)}`);
  }

  // 4d. Validation: half-specified parameters are an INVALID_REQUEST.
  const halfRes = await fetch(`${baseUrl}/api/v1/recover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...sample, nominalInterval: 10 }),
  });
  if (halfRes.status !== 400) fail(`half-specified budget returned HTTP ${halfRes.status}`);
  const halfBody = await halfRes.json();
  if (halfBody.error.code !== 'INVALID_REQUEST') fail('half-specified budget must be INVALID_REQUEST');

  // 4e. nominalInterval outside [minInterval, maxInterval] is INVALID_REQUEST.
  const oorRes = await fetch(`${baseUrl}/api/v1/recover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...sample, nominalInterval: 12, totalJitterBudget: 1 }),
  });
  if (oorRes.status !== 400) fail(`out-of-range nominalInterval returned HTTP ${oorRes.status}`);
  const oorBody = await oorRes.json();
  if (oorBody.error.code !== 'INVALID_REQUEST') fail('out-of-range nominalInterval must be INVALID_REQUEST');

  // 4f. Joint (not post-filtered) budget enforcement: fixed-time instance
  // whose unique timeline needs 2 cumulative jitter ticks. Budget 1 must be a
  // 422 with the first jitter blocker; budget 2 must succeed exhausted.
  const fixed = {
    modulus: 10,
    countLower: 0,
    countUpper: 100,
    minInterval: 1,
    maxInterval: 100,
    packets: [
      { id: 0, remainder: 0, timeLower: 0, timeUpper: 0 },
      { id: 1, remainder: 1, timeLower: 3, timeUpper: 3 },
      { id: 2, remainder: 2, timeLower: 6, timeUpper: 6 },
      { id: 3, remainder: 3, timeLower: 10, timeUpper: 10 },
      { id: 4, remainder: 4, timeLower: 12, timeUpper: 12 },
      { id: 5, remainder: 5, timeLower: 15, timeUpper: 15 },
    ],
  };
  const blockedRes = await fetch(`${baseUrl}/api/v1/recover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...fixed, nominalInterval: 3, totalJitterBudget: 1 }),
  });
  if (blockedRes.status !== 422) fail(`jitter-blocked instance returned HTTP ${blockedRes.status}`);
  const blockedBody = await blockedRes.json();
  if (blockedBody.error.code !== 'NO_CONSISTENT_INTERPRETATION') {
    fail('jitter-blocked instance must return NO_CONSISTENT_INTERPRETATION');
  }
  const jd = blockedBody.error.evidence?.detail;
  if (!jd || jd.cause !== 'JITTER_BUDGET' || !jd.jitterBudget) {
    fail(`jitter blocker evidence missing: ${JSON.stringify(blockedBody.error.evidence)}`);
  }
  if (jd.jitterBudget.budget !== 1 || typeof jd.jitterBudget.used !== 'number' ||
      jd.jitterBudget.minAdditionalJitter < 1 ||
      !Array.isArray(jd.jitterBudget.minimalByGap) || jd.jitterBudget.minimalByGap.length === 0) {
    fail(`jitter blocker numbers wrong: ${JSON.stringify(jd.jitterBudget)}`);
  }

  const exactRes = await fetch(`${baseUrl}/api/v1/recover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...fixed, nominalInterval: 3, totalJitterBudget: 2 }),
  });
  if (exactRes.status !== 200) fail(`exactly-exhausted budget returned HTTP ${exactRes.status}`);
  const exactBody = await exactRes.json();
  if (exactBody.data.jitter.used !== 2 || exactBody.data.jitter.exhausted !== true) {
    fail(`exactly-exhausted budget summary wrong: ${JSON.stringify(exactBody.data.jitter)}`);
  }

  console.log('SMOKE PASSED');
  console.log(`  order     : ${data.order.join(' -> ')}`);
  console.log(`  counts    : ${counts.join(', ')}`);
  console.log(`  missing   : ${data.missingCountTotal} packets in ${segments.length} segment(s)`);
  console.log(`  adjacency : all ${data.adjacency.length} constraints satisfied`);
  console.log(`  jitter    : budget 0/42 honored, budget 1 blocked with evidence, budget 2 exhausted`);
}

main().catch((err) => fail(err.stack ?? String(err)));
