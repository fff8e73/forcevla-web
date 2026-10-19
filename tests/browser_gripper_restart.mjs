// Fake gripper and fake HOME backends; execution worker dispatch is prohibited.
import assert from 'node:assert/strict';import {chromium} from '@playwright/test';import {writeFile} from 'node:fs/promises';import {setTimeout as delay} from 'node:timers/promises';
const base='http://127.0.0.1:8108',out=process.env.GRIPPER_TEST_OUTPUT||'artifacts/gripper-restart-ui-20261015';
const browser=await chromium.launch({headless:true,executablePath:'/home/user/.cache/ms-playwright/chromium-1161/chrome-linux/chrome'});
const report={hardware_commands_sent:0};
try{
 const context=await browser.newContext({viewport:{width:1440,height:1100}});const page=await context.newPage();const errors=[],posts=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.method()==='POST')posts.push({url:r.url(),body:r.postDataJSON()});});
 await page.goto(base,{waitUntil:'domcontentloaded'});for(let i=0;i<50&&await page.locator('#restart-gripper').isDisabled();i++)await delay(100);
 assert.ok(await page.locator('#restart-gripper').isEnabled());assert.equal(posts.length,0);
 await page.locator('#restart-gripper').click();assert.ok(await page.locator('#restart-gripper').isDisabled());assert.ok(await page.locator('#home').isDisabled());assert.ok(await page.locator('#continuous').isDisabled());assert.ok(await page.locator('#single').isDisabled());
 let run;for(let i=0;i<80;i++){run=(await(await context.request.get(base+'/api/v1/gripper-restart/status')).json()).run;if(run?.state==='COMPLETED')break;await delay(100);}
 assert.equal(run.state,'COMPLETED');assert.equal(run.receipt.goal_readback_confirmed,true);assert.equal(run.physical_arrival_verified,false);assert.deepEqual(run.after.applied_target,[0,0]);assert.deepEqual(run.after.actual,[-.0211,.79144]);
 for(let i=0;i<30&&await page.locator('#restart-gripper').isDisabled();i++)await delay(100);
 assert.ok(await page.locator('#home').isEnabled());assert.ok(await page.locator('#continuous').isEnabled());assert.ok(await page.locator('#restart-gripper').isEnabled());assert.match(await page.locator('#restart-gripper-status').textContent(),/初始目标已下发并确认/);
 assert.equal(posts.length,1);assert.ok(posts[0].url.endsWith('/gripper-restart/start'));
 await page.locator('.task-panel').screenshot({path:out+'/desktop.png'});report.layouts=[];
 for(const width of [390,320]){
  await page.setViewportSize({width,height:844});await delay(100);
  const boxes=await page.locator('.task-panel .actions > button').evaluateAll(els=>els.map(e=>{const r=e.getBoundingClientRect();return {id:e.id,x:r.x,y:r.y,width:r.width,height:r.height,overflow:e.scrollWidth>e.clientWidth};}));
  assert.equal(boxes.length,6);assert.equal(boxes[4].y,boxes[5].y);assert.ok(Math.abs(boxes[4].width-boxes[5].width)<1);assert.ok(boxes.every(b=>!b.overflow));report.layouts.push({width,boxes});
  await page.locator('.task-panel').screenshot({path:`${out}/mobile-${width}.png`});
 }
 assert.deepEqual(errors,[]);report.status='passed';report.requests=posts;report.restart_result=run;report.other_starts_blocked_during_restart=true;report.no_automatic_restart_on_load=true;report.browser_errors=errors;console.log(JSON.stringify({status:report.status,hardware_commands_sent:0,requests:posts.length,after:run.after.applied_target,mobile_layouts:report.layouts.length}));
}finally{await writeFile(out+'/browser-report.json',JSON.stringify(report,null,2)+'\n');await browser.close();}
