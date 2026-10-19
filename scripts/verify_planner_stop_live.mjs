// Explicit small live stop verification; excluded from automated tests.
import {chromium} from '@playwright/test';
import {mkdir,writeFile} from 'node:fs/promises';
import {setTimeout as delay} from 'node:timers/promises';
if(process.env.PLANNER_LIVE_TEST!=='1')throw Error('Explicit hardware authorization required');
const base='http://172.30.1.208:8096',folder='artifacts/planner-loop-stop-live';
await mkdir(folder,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:'/home/user/.cache/ms-playwright/chromium-1161/chrome-linux/chrome'});
const context=await browser.newContext();const page=await context.newPage();
let id=null,done=false;const terminal=['CANCELLED','FAILED','TIMEOUT','STOP_UNCONFIRMED','COMPLETED_LIMIT','INTERRUPTED_REVIEW'];
try{
 await page.goto(base,{waitUntil:'domcontentloaded'});
 await page.locator('#cl-mode').selectOption('planner');await page.locator('#cl-cycles').fill('2');await page.locator('#cl-duration').fill('120');await page.locator('#seed').fill('0');
 const responsePromise=page.waitForResponse(r=>r.url().endsWith('/api/v1/closed-loop/runs')&&r.request().method()==='POST');
 await page.locator('#continuous').click({timeout:20000});const response=await responsePromise;const run=await response.json();
 if(response.status()!==202)throw Error(JSON.stringify(run));id=run.id;console.log('STARTED',id);
 let clicked=false,clickedAt=null,runAtClick=null,result;const end=Date.now()+135000;
 while(Date.now()<end){
  result=await(await context.request.get(base+'/api/v1/closed-loop/runs/'+id)).json();
  if(terminal.includes(result.state)){done=true;break;}
  if(!clicked&&result.state==='EXECUTE'&&result.execution?.executed_model_steps>=1){
   runAtClick=result;clickedAt=Date.now();await page.locator('#execute20-stop').click();clicked=true;console.log('STOP_CLICKED',result.execution.executed_model_steps);
  }
  await delay(100);
 }
 const report={stop_clicked:clicked,click_time:clickedAt,run_at_click:runAtClick,final:result,stop_to_confirmed_ms:clickedAt?Date.now()-clickedAt:null};
 await writeFile(folder+'/report.json',JSON.stringify(report,null,2)+'\n');
 await page.screenshot({path:folder+'/page.png',fullPage:true});
 console.log(JSON.stringify({id,state:result?.state,steps:result?.execution?.executed_model_steps,stop:result?.stop?.confirmed,clicked}));
 if(!done||!clicked||result.state!=='CANCELLED'||!result.stop?.confirmed)process.exitCode=2;
}finally{
 if(id&&!done)await page.evaluate(async id=>{const s=await(await fetch('/api/v1/session')).json();await fetch('/api/v1/closed-loop/runs/'+id+'/stop',{method:'POST',headers:{'X-CSRF-Token':s.csrf_token}});},id).catch(()=>{});
 await browser.close();
}
