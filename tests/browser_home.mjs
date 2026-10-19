// This fixture injects a fake Apex backend and prohibits execution worker dispatch.
import assert from 'node:assert/strict';
import {chromium} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
import {setTimeout as delay} from 'node:timers/promises';
const base='http://127.0.0.1:8107',out='artifacts/home-ui-20261015';
const browser=await chromium.launch({headless:true,executablePath:'/home/user/.cache/ms-playwright/chromium-1161/chrome-linux/chrome'});
const report={hardware_commands_sent:0};
try{
 const context=await browser.newContext({viewport:{width:1440,height:1100}});const page=await context.newPage();const errors=[],posts=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.method()==='POST')posts.push({url:r.url(),body:r.postDataJSON()});});
 await page.goto(base,{waitUntil:'domcontentloaded'});for(let i=0;i<40&&await page.locator('#home').isDisabled();i++)await delay(100);
 assert.ok(await page.locator('#home').isEnabled());assert.equal(posts.length,0);
 await page.locator('#home').click();
 for(let i=0;i<30;i++){if(await page.locator('#home').isDisabled())break;await delay(50);}
 assert.ok(await page.locator('#home').isDisabled());assert.ok(await page.locator('#single').isDisabled());assert.ok(await page.locator('#continuous').isDisabled());
 let run;for(let i=0;i<80;i++){run=(await(await context.request.get(base+'/api/v1/home/status')).json()).run;if(run?.state==='COMPLETED')break;await delay(100);}
 assert.equal(run.state,'COMPLETED');assert.equal(run.final_robot.current_input_mode,0);assert.equal(run.native_result.success,true);
 for(let i=0;i<30&&await page.locator('#home').isDisabled();i++)await delay(100);
 assert.ok(await page.locator('#home').isEnabled());assert.ok(await page.locator('#continuous').isEnabled());assert.match(await page.locator('#home-status').textContent(),/HOME复位完成/);
 assert.equal(posts.filter(p=>p.url.endsWith('/home/start')).length,1);assert.ok(!posts.some(p=>p.url.endsWith('/closed-loop/runs')));
 await page.locator('.task-panel').screenshot({path:out+'/desktop.png'});
 await page.setViewportSize({width:390,height:844});await delay(200);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await page.locator('.task-panel').screenshot({path:out+'/mobile.png'});
 assert.deepEqual(errors,[]);report.status='passed';report.requests=posts;report.home_result=run;report.blocks_starts_during_home=true;report.no_automatic_reset_on_load=true;report.mobile_passed=true;report.browser_errors=errors;console.log(JSON.stringify(report,null,2));
}finally{await writeFile(out+'/browser-report.json',JSON.stringify(report,null,2)+'\n');await browser.close();}
