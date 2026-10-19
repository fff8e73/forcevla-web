import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
await mkdir("artifacts/closed-loop", { recursive: true });
const browser = await chromium.launch({
  headless: true,
  executablePath:
    process.env.CHROMIUM_PATH ||
    "/home/user/.cache/ms-playwright/chromium-1161/chrome-linux/chrome",
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
const url = process.env.LOOP_TEST_URL || "http://127.0.0.1:8101";
try {
  await page.goto(url);
  await page.waitForFunction(
    () => !document.querySelector("#cl-start").disabled,
  );
  await page.locator("#cl-start").click();
  await page.waitForFunction(
    () => document.querySelector("#cl-progress").textContent.includes("3 / 3"),
    { timeout: 30000 },
  );
  await page.waitForFunction(() =>
    document.querySelector("#cl-state").textContent.includes("已完成指定轮数"),
  );
  assert.match(await page.locator("#cl-result").textContent(), /模拟数据/);
  assert.match(await page.locator("#cl-hold").textContent(), /最终保持已确认/);
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#cl-export").click(),
  ]);
  await download.saveAs("artifacts/closed-loop/browser-export.zip");
  await page.locator("#cl-cycles").fill("10");
  await page.locator("#cl-start").click();
  await page.waitForFunction(() =>
    document.querySelector("#cl-state").textContent.includes("推理中"),
  );
  const before = await page.evaluate(
    async () =>
      (await (await fetch("/api/v1/closed-loop/status")).json()).run.id,
  );
  await page.reload();
  await page.waitForFunction(
    () => !document.querySelector("#cl-stop").disabled,
  );
  const after = await page.evaluate(
    async () =>
      (await (await fetch("/api/v1/closed-loop/status")).json()).run.id,
  );
  assert.equal(before, after);
  await page.locator("#cl-stop").click();
  await page.waitForFunction(
    () => document.querySelector("#cl-state").textContent.includes("已取消"),
    { timeout: 10000 },
  );
  await page.locator("#cl-refresh").click();
  await page.waitForFunction(
    () => document.querySelector("#cl-history").options.length >= 3,
  );
  await page.locator("#cl-history").selectOption({ index: 2 });
  await page.waitForFunction(() =>
    document.querySelector("#cl-result").textContent.includes("历史"),
  );
  await page.screenshot({
    path: "artifacts/closed-loop/desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
    false,
  );
  await page.screenshot({
    path: "artifacts/closed-loop/mobile.png",
    fullPage: true,
  });
  assert.deepEqual(errors, []);
  await writeFile(
    "artifacts/closed-loop/browser-report.json",
    JSON.stringify(
      {
        url,
        errors,
        threeCycles: true,
        reloadSameRun: true,
        cancel: true,
        export: true,
        history: true,
        mobile: true,
      },
      null,
      2,
    ),
  );
  console.log("Closed-loop browser checks passed");
} finally {
  await browser.close();
}
