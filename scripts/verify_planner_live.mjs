// Explicit live hardware verification. Never part of the automatic test suite.
import { chromium } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import { setTimeout as delay } from 'node:timers/promises';
if(process.env.PLANNER_LIVE_TEST !== '1') throw Error('Set PLANNER_LIVE_TEST=1 to explicitly execute hardware');
const target=process.env.PLANNER_TEST_URL || 'http://172.30.1.208:8096';
const output=process.env.PLANNER_TEST_OUTPUT || 'artifacts/planner-loop-live';
await mkdir(output,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:'/home/user/.cache/ms-playwright/chromium-1161/chrome-linux/chrome'});
const context=await browser.newContext({viewport:{width:1440,height:1100}});
const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
const terminal=new Set(['COMPLETED_LIMIT','CANCELLED','FAILED','TIMEOUT','STOP_UNCONFIRMED','INTERRUPTED_REVIEW']);
let runId=null,finished=false;
try {
 await page.goto(target,{waitUntil:'domcontentloaded'});
 await page.locator('#cl-mode').selectOption('planner');
 await page.locator('#cl-cycles').fill('2');
 await page.locator('#cl-duration').fill('180');
 await page.locator('#seed').fill('0');
 const responsePromise=page.waitForResponse(r=>r.request().method()==='POST' && r.url().endsWith('/api/v1/closed-loop/runs'));
 await page.locator('#continuous').click({timeout:20000});
 const response=await responsePromise;const started=await response.json();
 if(response.status()!==202)throw Error(JSON.stringify(started));
 runId=started.id;console.log('STARTED',runId);
 const end=Date.now()+195000;let last='';let value;
 while(Date.now()<end){
  const r=await context.request.get(target+'/api/v1/closed-loop/runs/'+runId);value=await r.json();
  const label=`${value.state} cycle=${value.cycle} steps=${value.execution?.executed_model_steps||0} completed=${value.completed_cycles}`;
  if(label!==last){console.log(label);last=label;}
  if(terminal.has(value.state)){finished=true;break;}
  await delay(200);
 }
 if(!finished)throw Error('Live verification did not terminate');
 await writeFile(output+'/report.json',JSON.stringify({run:value,browser_errors:errors},null,2)+'\n');
 await page.screenshot({path:output+'/page.png',fullPage:true});
 console.log('RESULT',JSON.stringify({id:runId,state:value.state,completed_cycles:value.completed_cycles,reason:value.reason,stop:value.stop}));
 if(value.state!=='COMPLETED_LIMIT'||value.completed_cycles!==2)process.exitCode=2;
} finally {
 if(runId&&!finished){
  await page.evaluate(async id=>{
   const session=await(await fetch('/api/v1/session')).json();
   await fetch('/api/v1/closed-loop/runs/'+id+'/stop',{method:'POST',headers:{'X-CSRF-Token':session.csrf_token}});
  },runId).catch(()=>{});
 }
 await browser.close();
}
