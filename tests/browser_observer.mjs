import { chromium } from '@playwright/test';
import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';

const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||'/home/user/.cache/ms-playwright/chromium-1161/chrome-linux/chrome'});
try {
  const context=await browser.newContext({viewport:{width:1440,height:1100}});
  const page=await context.newPage();
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8096');
  await page.waitForFunction(()=>document.querySelector('#connection').textContent.includes('已连接'));
  await page.waitForFunction(()=>document.querySelectorAll('.camera img:not([hidden])').length===4);
  await page.waitForFunction(()=>!document.querySelector('#trajectory-empty').offsetParent);
  const state=await page.evaluate(async()=> (await (await fetch('/api/v1/status')).json()));
  assert.ok(state.run && ['RUNNING','WARMING','WAITING_INPUT'].includes(state.run.state));
  assert.equal(state.run.is_owner,false);
  assert.equal(await page.locator('#stop').isDisabled(),true);
  assert.equal(await page.locator('#single').isDisabled(),true);
  const denied=await page.evaluate(async(id)=>{
    const session=await (await fetch('/api/v1/session')).json();
    const response=await fetch(`/api/v1/runs/${id}/stop`,{method:'POST',headers:{'X-CSRF-Token':session.csrf_token}});
    return response.status;
  },state.run.id);
  assert.equal(denied,409);
  await page.screenshot({path:'artifacts/stability-20261014/observer-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.waitForTimeout(350);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:'artifacts/stability-20261014/observer-mobile.png',fullPage:true});
  await page.reload();
  await page.waitForFunction(()=>document.querySelector('#connection').textContent.includes('已连接'));
  const after=await page.evaluate(async()=> (await (await fetch('/api/v1/status')).json()));
  assert.equal(after.run.id,state.run.id);
  assert.equal(after.run.is_owner,false);
  assert.deepEqual(errors,[]);
  const report={source:'real live service, viewer only',four_images:true,trajectory:true,foreign_stop_status:denied,owner_run_unchanged:true,reload:true,mobile_no_overflow:true,errors,passed:true};
  await writeFile('artifacts/stability-20261014/browser-observer.json',JSON.stringify(report,null,2)+'\n');
  console.log(report);
} finally {await browser.close()}
