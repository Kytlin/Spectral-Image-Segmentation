import { test } from "node:test";
import assert from "node:assert/strict";
import { createLatest } from "./latest.js";

const delay = (ms, value, signal) =>
  new Promise((resolve, reject) => {
    const t = setTimeout(() => resolve(value), ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
    });
  });

test("a slow older response cannot overwrite a newer one", async () => {
  const latest = createLatest();
  // The old task ignores abort, as a fetch whose response is already in flight might.
  const old = latest.run(() => delay(50, "old"));
  const fresh = latest.run(() => delay(5, "new"));
  assert.deepEqual(await fresh, { current: true, value: "new" });
  assert.deepEqual(await old, { current: false });
});

test("starting a task aborts the previous one", async () => {
  const latest = createLatest();
  let aborted = false;
  const first = latest.run((signal) => {
    signal.addEventListener("abort", () => (aborted = true));
    return delay(50, "first", signal);
  });
  await latest.run(() => delay(1, "second"));
  assert.equal(aborted, true);
  assert.deepEqual(await first, { current: false });
});

test("invalidate supersedes the running task", async () => {
  const latest = createLatest();
  const pending = latest.run(() => delay(10, "x"));
  latest.invalidate();
  assert.deepEqual(await pending, { current: false });
});

test("errors of the current task propagate; errors of stale tasks do not", async () => {
  const latest = createLatest();
  const stale = latest.run(async () => {
    await delay(10);
    throw new Error("stale failure");
  });
  const current = latest.run(async () => {
    throw new Error("current failure");
  });
  await assert.rejects(current, /current failure/);
  assert.deepEqual(await stale, { current: false });
});
