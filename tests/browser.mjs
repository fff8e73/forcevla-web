import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
await mkdir("artifacts", { recursive: true });
const browser = await chromium.launch({
  headless: true,
  executablePath:
    process.env.CHROMIUM_PATH ||
    "/home/user/.cache/ms-playwright/chromium-1161/chrome-linux/chrome",
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
const url = process.env.FORCEVLA_TEST_URL || "http://127.0.0.1:8097";
try {
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() =>
    document.querySelector("#connection").textContent.includes("已连接"),
  );
  await page
    .locator("#prompt")
    .fill(
      "Insert the Ethernet cable into the Ethernet port on the end face of the docking station.",
    );
  await page.locator("#single").click();
  await page.waitForFunction(
    () =>
      document.querySelector("#prediction-id").textContent.includes("预测 ") &&
      !document.querySelector("#prediction-id").textContent.includes("尚无"),
    { timeout: 30000 },
  );
  await page.waitForFunction(
    () => document.querySelectorAll(".camera img:not([hidden])").length === 4,
  );
  assert.equal(await page.locator("#trajectory-empty").isVisible(), false);
  await page.locator("#view-snapshot").click();
  await page.waitForFunction(() =>
    document.querySelector("#camera-mode").textContent.includes("固定帧"),
  );
  await page.locator("#step").fill("30");
  await page.locator("#step").dispatchEvent("input");
  assert.match(await page.locator("#step-label").textContent(), /31 \/ 50/);
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#export").click(),
  ]);
  await download.saveAs("artifacts/browser-export.zip");
  await page.screenshot({ path: "artifacts/desktop.png", fullPage: true });
  await page.locator("#single").click();
  await page.waitForFunction(() =>
    ["RUNNING", "WARMING"].includes(
      document.querySelector("#run-status").textContent,
    ),
  );
  await page.locator("#stop").click();
  await page.waitForFunction(
    () => document.querySelector("#run-status").textContent === "IDLE",
  );
  await page.screenshot({ path: "artifacts/stopped.png", fullPage: true });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
    false,
  );
  await page.reload();
  await page.waitForFunction(() =>
    document.querySelector("#connection").textContent.includes("已连接"),
  );
  await page.locator("#refresh-history").click();
  assert.ok((await page.locator("#history-run option").count()) > 1);
  await page.locator("#history-run").selectOption({ index: 2 });
  await page.waitForFunction(
    () => document.querySelector("#history-prediction").options.length > 0,
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(300);
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
    false,
  );
  await page.screenshot({ path: "artifacts/mobile.png", fullPage: true });
  assert.deepEqual(errors, []);
  await writeFile(
    "artifacts/browser-report.json",
    JSON.stringify(
      {
        url,
        errors,
        single: true,
        singleStop: true,
        snapshot: true,
        export: true,
        reload: true,
        mobile: true,
      },
      null,
      2,
    ),
  );
  console.log(
    "Browser checks passed: single, single/stop, snapshot, export, reload/history, mobile.",
  );
} finally {
  await browser.close();
}
