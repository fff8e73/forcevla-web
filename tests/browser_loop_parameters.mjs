// Uses a server fixture that prohibits all execution, plus intercepts start requests.
import assert from 'node:assert/strict';
import {chromium} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
import {setTimeout as delay} from 'node:timers/promises';
const base='http://127.0.0.1:8105';
const browser=await chromium.launch({headless:true,executablePath:'/home/user/.cache/ms-playwright/chromium-1161/chrome-linux/chrome'});
const report={hardware_commands_sent:0,requests:[]};
try{
 const page=await browser.newPage({viewport:{width:1440,height:1100}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/api/v1/closed-loop/runs',async route=>{
  if(route.request().method()!=='POST')return route.continue();
  const options=route.request().postDataJSON();report.requests.push(options);
  await route.fulfill({status:202,contentType:'application/json',body:JSON.stringify({id:'a'.repeat(32),mode:'live',engine:'movej',state:'COMPLETED_LIMIT',is_owner:true,completed_cycles:options.max_cycles,config:options})});
 });
 await page.goto(base,{waitUntil:'domcontentloaded'});
 for(let i=0;i<60&&await page.locator('#continuous').isDisabled();i++)await delay(100);
 assert.equal(await page.locator('#cl-mode').inputValue(),'planner');
 assert.equal(await page.locator('#cl-cycles').inputValue(),'50');assert.equal(await page.locator('#cl-k').inputValue(),'20');
 assert.ok(await page.locator('#cl-k').isEnabled());assert.equal(await page.locator('#execute20').count(),0);report.defaults={cycles:50,steps:20};
 await page.locator('#continuous').click();await delay(300);
 assert.equal(report.requests[0].max_cycles,50);assert.equal(report.requests[0].execution_horizon,20);
 await page.locator('#cl-k').fill('7');await page.locator('#cl-cycles').fill('3');
 assert.equal((await page.locator('#continuous').textContent()).trim(),'开始连续');assert.match(await page.locator('#continuous').getAttribute('title'),/7步/);assert.ok(await page.locator('#cl-start').isHidden());
 await page.locator('#continuous').click();await delay(300);
 assert.equal(report.requests[1].execution_horizon,7);assert.equal(report.requests[1].max_cycles,3);
 for(const bad of ['', '0','51','1.5']){
  await page.locator('#cl-k').fill(bad);assert.ok(await page.locator('#cl-start').isDisabled());assert.ok(await page.locator('#continuous').isDisabled());
 }
 await page.locator('#cl-k').fill('50');await page.locator('#cl-cycles').fill('101');assert.ok(await page.locator('#cl-start').isDisabled());
 await page.locator('#cl-cycles').fill('1');assert.ok(await page.locator('#cl-start').isEnabled());await page.locator('#continuous').click();await delay(300);
 assert.equal(report.requests[2].execution_horizon,50);assert.equal(report.requests[2].max_cycles,1);
 await page.reload();for(let i=0;i<40&&(await page.locator('#cl-mode').inputValue())!=='planner';i++)await delay(100);
 assert.equal(await page.locator('#cl-k').inputValue(),'20');assert.equal(await page.locator('#cl-cycles').inputValue(),'50');
 await page.screenshot({path:'artifacts/configurable-loop-20261015/desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});await delay(200);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 await page.screenshot({path:'artifacts/configurable-loop-20261015/mobile.png',fullPage:true});assert.deepEqual(errors,[]);
 report.status='passed';report.invalid_values_blocked=true;report.defaults_after_reload=true;report.browser_errors=errors;
 console.log(JSON.stringify(report,null,2));
}finally{
 await writeFile('artifacts/configurable-loop-20261015/browser-report.json',JSON.stringify(report,null,2)+'\n');await browser.close();
}
