// Read-only/synthetic fixture only. Never sends commands to the live robot service.
import assert from 'node:assert/strict';
import { chromium } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import { setTimeout as delay } from 'node:timers/promises';
const base='http://127.0.0.1:8104',out='artifacts/server-heartbeat-20261015';
await mkdir(out,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:'/home/user/.cache/ms-playwright/chromium-1161/chrome-linux/chrome'});
const result={synthetic_only:true,hardware_commands_sent:0};
try{
 const context=await browser.newContext();
 await context.addInitScript(()=>{
  window.setInterval=()=>0;
  Object.defineProperty(document,'visibilityState',{get:()=> 'hidden'});
  const Original=window.WebSocket;window.__sockets=[];
  window.WebSocket=class extends Original {constructor(...args){super(...args);window.__sockets.push(this);}};
 });
 const page=await context.newPage();const errors=[];let received=0;const acknowledgements=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('websocket',ws=>{ws.on('framereceived',()=>received++);ws.on('framesent',e=>{try{const v=JSON.parse(e.payload);if(v.kind==='heartbeat_ack')acknowledgements.push({time:Date.now(),sequence:v.sequence,connection_id:v.connection_id});}catch{}});});
 await page.goto(base,{waitUntil:'domcontentloaded'});
 await page.locator('#cl-mode').selectOption('simulate');await page.locator('#cl-cycles').fill('100');await page.locator('#cl-duration').fill('90');
 const responsePromise=page.waitForResponse(r=>r.url().endsWith('/api/v1/closed-loop/runs')&&r.request().method()==='POST');await page.locator('#cl-start').click();const response=await responsePromise;const started=await response.json();assert.equal(response.status(),202);
 const getRun=async()=>await(await context.request.get(base+'/api/v1/closed-loop/runs/'+started.id)).json();
 const getStatus=async()=>await(await context.request.get(base+'/api/v1/status')).json();
 const end=Date.now()+14000;
 while(Date.now()<end){await delay(250);const run=await getRun();assert.ok(['STARTING','HOLDING'].includes(run.state),JSON.stringify(run));}
 const healthy=await getStatus();assert.equal(healthy.connection_health.protocol,'server_challenge_v1');assert.equal(healthy.connection_health.visibility_state,'hidden');assert.ok(acknowledgements.length>=10);
 result.no_intervals_and_hidden_page={duration_s:14,acknowledgements:acknowledgements.length,run_state:(await getRun()).state,health:healthy.connection_health};
 const oldConnection=healthy.connection_health.connection_id;
 await page.evaluate(()=>window.__sockets.at(-1).close(1000,'diagnostic_reconnect'));
 let after;const reconnectEnd=Date.now()+4500;
 while(Date.now()<reconnectEnd){await delay(100);after=await getStatus();if(after.connection_health.connection_id!==oldConnection&&after.connection_health.protocol==='server_challenge_v1')break;}
 assert.notEqual(after.connection_health.connection_id,oldConnection);assert.equal(after.connection_health.protocol,'server_challenge_v1');assert.equal((await getRun()).state,'HOLDING');
 result.reconnect={same_run_id:started.id,old_connection:oldConnection,new_connection:after.connection_health.connection_id,run_state:(await getRun()).state};
 const framesBefore=received;const blockedAt=Date.now();
 await page.evaluate(()=>{const send=WebSocket.prototype.send;WebSocket.prototype.send=function(data){try{if(JSON.parse(data).kind==='heartbeat_ack')return;}catch{}return send.call(this,data);};});
 let stopped;const stopEnd=Date.now()+8500;
 while(Date.now()<stopEnd){await delay(100);stopped=await getRun();if(stopped.state==='CANCELLED')break;}
 assert.equal(stopped.state,'CANCELLED');assert.equal(stopped.reason,'browser_heartbeat_timeout');assert.ok(received-framesBefore>=10);assert.equal(stopped.connection_diagnostics.connection_health.state,'open');
 result.unresponsive_page={elapsed_ms:Date.now()-blockedAt,server_messages:received-framesBefore,state:stopped.state,reason:stopped.reason,diagnostics:stopped.connection_diagnostics};
 await page.reload();await delay(1800);assert.equal((await getRun()).state,'CANCELLED');result.reconnect_does_not_resume_cancelled_run=true;
 assert.deepEqual(errors,[]);result.browser_errors=errors;result.status='passed';
 await page.screenshot({path:out+'/browser.png',fullPage:true});
 console.log(JSON.stringify(result,null,2));
}finally{
 await writeFile(out+'/browser-report.json',JSON.stringify(result,null,2)+'\n');
 await browser.close();
}
