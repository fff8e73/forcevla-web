import { chromium } from '@playwright/test';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
await mkdir('artifacts/control-ui',{recursive:true});
const live=process.env.LIVE_READONLY==='1';
const url=process.env.CONTROL_TEST_URL || 'http://127.0.0.1:8098';
if (!live && url!=='http://127.0.0.1:8098') throw Error('Mutating browser test restricted to the local fake fixture');
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH || '/home/user/.cache/ms-playwright/chromium-1161/chrome-linux/chrome'});
const page=await browser.newPage({viewport:{width:1440,height:1100}});
const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));
page.on('request',r=>{if(r.method()==='POST')requests.push(new URL(r.url()).pathname);});
try {
  await page.goto(url);await page.waitForFunction(()=>document.querySelector('#connection').textContent.includes('已连接'));
  await page.locator('#cv-title').scrollIntoViewIfNeeded();
  assert.equal(await page.locator('#cv-cycle').isDisabled(),true);
  assert.equal(await page.locator('#cv-moving').isDisabled(),true);
  await page.locator('#cv-check').click();
  await page.waitForFunction(()=>!document.querySelector('#cv-check').disabled);
  if(!live){
    await page.waitForFunction(()=>document.querySelector('#cv-precheck').textContent.includes('只读条件满足'));
    await page.locator('#cv-safety').check();
    page.on('dialog',dialog=>dialog.accept());
    await page.locator('#cv-cycle').click();
    await page.waitForFunction(()=>document.querySelector('#cv-state').textContent.includes('静止链路验证完成'),{timeout:10000});
    assert.match(await page.locator('#cv-result').textContent(),/新运动目标：0/);
    assert.match(await page.locator('#cv-result').textContent(),/运动中停止：未验证/);
  }
  await page.screenshot({path:`artifacts/control-ui/${live?'live-readonly':'fake-cycle'}-desktop.png`,fullPage:true});
  await page.setViewportSize({width:390,height:844});await page.locator('#cv-title').scrollIntoViewIfNeeded();
  await page.waitForTimeout(250);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:`artifacts/control-ui/${live?'live-readonly':'fake-cycle'}-mobile.png`,fullPage:true});
  assert.deepEqual(errors,[]);
  if(live)assert.deepEqual(requests,['/api/v1/control-validation/check']);
  await writeFile(`artifacts/control-ui/${live?'live-readonly':'fake-cycle'}-report.json`,JSON.stringify({url,liveReadonly:live,errors,postPaths:requests,physicalCommandsTriggered:false},null,2));
  console.log('Control-validation browser checks passed:',live?'live read-only':'fake cycle');
} finally {await browser.close();}
