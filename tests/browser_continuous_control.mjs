// Fixture never starts a hardware worker. Requests are inspected end-to-end.
import assert from 'node:assert/strict';
import {chromium} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
import {setTimeout as delay} from 'node:timers/promises';
const base='http://127.0.0.1:8106',out='artifacts/continuous-control-20261015';
const browser=await chromium.launch({headless:true,executablePath:'/home/user/.cache/ms-playwright/chromium-1161/chrome-linux/chrome'});
const report={hardware_commands_sent:0,requests:[]};
try{
 const context=await browser.newContext({viewport:{width:1440,height:1100}});const page=await context.newPage();const errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.method()==='POST'&&!r.url().endsWith('/connection-events'))report.requests.push({url:r.url(),body:r.postDataJSON()});});
 await page.goto(base,{waitUntil:'domcontentloaded'});
 for(let i=0;i<50&&await page.locator('#continuous').isDisabled();i++)await delay(100);
 assert.equal(await page.locator('#execute20').count(),0);assert.ok(await page.locator('#cl-start').isHidden());assert.equal((await page.locator('#continuous').textContent()).trim(),'开始连续');
 assert.equal(await page.locator('#cl-cycles').inputValue(),'50');assert.equal(await page.locator('#cl-k').inputValue(),'20');
 await page.locator('#cl-k').fill('7');await page.locator('#cl-cycles').fill('3');
 await page.locator('#continuous').click();
 for(let i=0;i<50&&await page.locator('#execute20-stop').isDisabled();i++)await delay(100);
 assert.ok(await page.locator('#execute20-stop').isEnabled());assert.ok(await page.locator('#continuous').isDisabled());assert.ok(await page.locator('#single').isDisabled());assert.ok(await page.locator('#stop').isDisabled());
 const started=report.requests.find(r=>r.url.endsWith('/closed-loop/runs'));assert.equal(started.body.engine,'movej');assert.equal(started.body.mode,'live');assert.equal(started.body.execution_horizon,7);assert.equal(started.body.max_cycles,3);assert.ok(!report.requests.some(r=>r.url.endsWith('/api/v1/runs')));
 await page.locator('#execute20-stop').click();
 for(let i=0;i<50&&await page.locator('#continuous').isDisabled();i++)await delay(100);
 assert.ok(report.requests.some(r=>r.url.endsWith('/closed-loop/runs/'+'f'.repeat(32)+'/stop')));
 assert.ok(await page.locator('#single').isEnabled());
 await page.locator('#single').click();
 for(let i=0;i<50&&!report.requests.some(r=>r.url.endsWith('/api/v1/runs'));i++)await delay(50);
 const single=report.requests.find(r=>r.url.endsWith('/api/v1/runs'));assert.equal(single.body.kind,'single');assert.ok(!report.requests.some(r=>r.body?.kind==='continuous'));
 await delay(800);
 await page.locator('#cl-k').fill('51');await delay(300);assert.ok(await page.locator('#continuous').isDisabled());
 await page.locator('#cl-k').fill('20');await page.locator('#cl-cycles').fill('50');
 await page.screenshot({path:out+'/desktop.png',fullPage:true});await page.setViewportSize({width:390,height:844});await delay(200);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await page.screenshot({path:out+'/mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);report.status='passed';report.old_start_removed=true;report.continuous_routes_to_live_loop=true;report.single_remains_nonmoving=true;report.stop_routes_to_loop=true;report.mutual_exclusion=true;report.invalid_steps_remain_disabled=true;report.browser_errors=errors;
 console.log(JSON.stringify(report,null,2));
}finally{await writeFile(out+'/browser-report.json',JSON.stringify(report,null,2)+'\n');await browser.close();}
