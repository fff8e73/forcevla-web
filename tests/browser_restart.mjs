import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';

const root = process.cwd();
const env = { ...process.env };
delete env.PYTHONPATH;
delete env.VIRTUAL_ENV;
const url = 'http://127.0.0.1:8097';
let server, browser;
const errors=[];
async function start() {
  server=spawn(`${root}/.venv/bin/python`,['-m','forcevla_web','--demo','--port','8097'],{cwd:root,env,stdio:'ignore'});
  for(let i=0;i<60;i++) {
    if(server.exitCode!==null)throw Error(`Demo server exited: ${server.exitCode}`);
    try { if((await fetch(url)).ok)return; } catch {}
    await new Promise(r=>setTimeout(r,150));
  }
  throw Error('Demo server startup timeout');
}
async function stop(signal) {
  if(!server||server.exitCode!==null)return;
  const exit=new Promise(resolve=>server.once('exit',resolve));
  server.kill(signal);await exit;
}
try {
  await mkdir('artifacts/stability-20261014',{recursive:true});
  await start();
  browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||'/home/user/.cache/ms-playwright/chromium-1161/chrome-linux/chrome'});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url);
  await page.waitForFunction(()=>document.querySelector('#connection').textContent.includes('已连接'));
  await page.locator('#prompt').fill('Demo reconnect test');
  // Exercise the prediction API explicitly; the UI continuous button now moves hardware.
  await page.evaluate(async () => {
    const session = await (await fetch('/api/v1/session')).json();
    const response = await fetch('/api/v1/runs', {
      method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRF-Token': session.csrf_token},
      body: JSON.stringify({kind:'continuous',prompt:'Demo reconnect test',command_id:'demo-reconnect-'+Date.now()}),
    });
    if (!response.ok) throw Error('Demo prediction start failed');
  });
  await page.waitForFunction(()=>document.querySelector('#run-count').textContent.startsWith('1 次')||document.querySelector('#run-count').textContent.startsWith('2 次'));
  const run=await page.evaluate(async()=> (await (await fetch('/api/v1/status')).json()).run);
  const cookieBefore=(await page.context().cookies()).find(c=>c.name==='forcevla_session').value;
  const startTime=Date.now();
  await stop('SIGKILL');
  await page.waitForFunction(()=>document.querySelector('#connection').textContent.includes('连接中断'));
  await start();
  await page.waitForFunction(()=>document.querySelector('#connection').textContent.includes('已连接'),{timeout:15000});
  const recovered=await page.evaluate(async()=> (await (await fetch('/api/v1/status')).json()));
  assert.equal(recovered.run,null);
  const history=await page.evaluate(async(id)=> (await (await fetch(`/api/v1/runs/${id}`)).json()),run.id);
  assert.equal(history.state,'INTERRUPTED');
  const cookieAfter=(await page.context().cookies()).find(c=>c.name==='forcevla_session').value;
  assert.ok(cookieBefore===cookieAfter,"Session cookie was not preserved");
  await page.locator('#single').click();
  await page.waitForFunction(()=>document.querySelector('#run-status').textContent==='IDLE' && document.querySelector('#run-count').textContent.startsWith('1 次'));
  assert.deepEqual(errors,[]);
  const report={source:'demo only',abrupt_restart:true,session_cookie_preserved:true,old_run_interrupted:true,automatic_inference_resume:false,manual_new_inference:true,reconnect_and_check_s:(Date.now()-startTime)/1000,errors,passed:true};
  await writeFile('artifacts/stability-20261014/browser-restart.json',JSON.stringify(report,null,2)+'\n');
  console.log(report);
} finally {
  if(browser)await browser.close();
  await stop('SIGINT');
}
