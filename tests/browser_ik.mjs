import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { writeFile, mkdir } from "node:fs/promises";
const browser = await chromium.launch({
  headless: true,
  executablePath:
    process.env.CHROMIUM_PATH ||
    "/home/user/.cache/ms-playwright/chromium-1161/chrome-linux/chrome",
});
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1100 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("http://127.0.0.1:8096");
  await page.waitForFunction(() =>
    document.querySelector("#connection").textContent.includes("已连接"),
  );
  await page.waitForFunction(() =>
    document.querySelector("#ik-service").textContent.includes("在线"),
  );
  await page.locator("#single").click();
  await page.waitForFunction(
    () => {
      const t = document.querySelector("#ik-summary").textContent;
      return (
        t.includes("通过") ||
        t.includes("未完成") ||
        t.includes("起点") ||
        t.includes("错误") ||
        t.includes("不可用")
      );
    },
    { timeout: 30000 },
  );
  const result = await page.evaluate(async () => {
    const s = await (await fetch("/api/v1/status")).json();
    return await (
      await fetch(`/api/v1/predictions/${s.latest_prediction_id}`)
    ).json();
  });
  assert.equal(typeof result.offline_ik, "object");
  assert.equal(
    result.offline_ik.status,
    "complete",
    JSON.stringify(result.offline_ik),
  );
  assert.equal(result.offline_ik.checked_steps, 50);
  assert.equal(result.offline_ik.model.collision_pair_count, 8);
  assert.equal(result.offline_ik.execution_verified, false);
  assert.equal(await page.locator("#ik-joints tr").count(), 7);
  const before = await page.locator("#joint-pose").screenshot();
  await page.locator("#step").fill("49");
  await page.locator("#step").dispatchEvent("input");
  assert.match(await page.locator("#ik-step-info").textContent(), /第 50 步/);
  await page.locator("#projection").selectOption("xz");
  assert.notDeepEqual(before, await page.locator("#joint-pose").screenshot());
  await mkdir("artifacts/offline-ik-20261014", { recursive: true });
  await page.screenshot({
    path: "artifacts/offline-ik-20261014/desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(300);
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
    false,
  );
  await page.screenshot({
    path: "artifacts/offline-ik-20261014/mobile.png",
    fullPage: true,
  });
  assert.deepEqual(errors, []);
  const report = {
    passed: true,
    prediction_id: result.id,
    ik_status: result.offline_ik.status,
    ik_passed: result.offline_ik.passed,
    checked_steps: result.offline_ik.checked_steps,
    reasons: result.offline_ik.reasons,
    elapsed_s: result.offline_ik.elapsed_s,
    seed_fk_residual: result.offline_ik.seed_fk_residual,
    errors,
    mobile: true,
  };
  await writeFile(
    "artifacts/offline-ik-20261014/browser-report.json",
    JSON.stringify(report, null, 2) + "\n",
  );
  console.log(report);
} finally {
  await browser.close();
}
