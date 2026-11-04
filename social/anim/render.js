#!/usr/bin/env node
// node render.js [names...] [--stills 1,5,9] [--gif] [--master] [--only-master] [--fps 30]
// --master: after rendering, concat intro,matvec,adder,stack,hardware,outro into out/explainers_reel.mp4 (--only-master reuses existing segments)
// Renders social/anim/<name>.html frame by frame (headless Chrome via puppeteer-core -> ffmpeg).
const fs=require('fs'),path=require('path'),{spawn,spawnSync}=require('child_process');
const puppeteer=require(path.resolve(__dirname,'../video/node_modules/puppeteer-core'));
const args=process.argv.slice(2);const flag=k=>args.includes('--'+k);const val=k=>{const i=args.indexOf('--'+k);return i<0?null:args[i+1]};
const OUT=path.join(__dirname,'out');fs.mkdirSync(path.join(OUT,'stills'),{recursive:true});
const CHROME=process.env.CHROME||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const FPS=Number(val('fps')||30);
const ALL={dashboard:'dashboard.html',matvec:'matvec.html',adder:'adder.html',stack:'stack.html',hardware:'hardware.html',intro:'titles.html?scene=intro',outro:'titles.html?scene=outro'};
const skip=new Set(['--stills','--fps']);
const names=args.filter((a,i)=>!a.startsWith('--')&&!skip.has(args[i-1]));
const todo=names.length?names:Object.keys(ALL);
async function open(browser,name){const p=await browser.newPage();await p.setViewport({width:1920,height:1080,deviceScaleFactor:1});
 await p.goto('file://'+path.join(__dirname,ALL[name]));await p.waitForFunction('window.READY===true');return p}
async function renderOne(browser,name){
 const p=await open(browser,name);const dur=await p.evaluate(()=>window.DUR);const n=Math.round(dur*FPS);
 const file=path.join(OUT,name+'.mp4');
 const ff=spawn('ffmpeg',['-y','-loglevel','error','-f','image2pipe','-framerate',String(FPS),'-c:v','mjpeg','-i','-','-c:v','libx264','-preset','medium','-crf','17','-pix_fmt','yuv420p','-r',String(FPS),'-movflags','+faststart',file],{stdio:['pipe','inherit','inherit']});
 const done=new Promise(r=>ff.on('close',r));
 for(let i=0;i<n;i++){await p.evaluate(t=>window.render(t),i/FPS);const b=await p.screenshot({type:'jpeg',quality:95});if(!ff.stdin.write(b))await new Promise(r=>ff.stdin.once('drain',r))}
 ff.stdin.end();await done;await p.close();console.log('wrote',file,dur+'s');
 if(flag('gif')){const g=path.join(OUT,name+'.gif');spawnSync('ffmpeg',['-y','-loglevel','error','-i',file,'-vf','fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4','-loop','0',g]);console.log('wrote',g,(fs.statSync(g).size/1e6).toFixed(1)+' MB')}
}
const ORDER=['intro','matvec','adder','stack','hardware','outro'];
function master(){ // concat the rendered segments with a short fade-in on each (needs out/<name>.mp4 for all of ORDER)
 const inputs=ORDER.flatMap(n=>['-i',path.join(OUT,n+'.mp4')]);
 const f=ORDER.map((n,i)=>`[${i}:v]${i?'fade=t=in:st=0:d=0.35,':''}setsar=1[v${i}]`).join(';')+';'+ORDER.map((n,i)=>`[v${i}]`).join('')+`concat=n=${ORDER.length}:v=1:a=0[v]`;
 const out=path.join(OUT,'explainers_reel.mp4');
 const r=spawnSync('ffmpeg',['-y','-loglevel','error',...inputs,'-filter_complex',f,'-map','[v]','-c:v','libx264','-preset','slow','-crf','18','-pix_fmt','yuv420p','-r','30','-movflags','+faststart',out],{stdio:'inherit'});
 console.log('wrote',out,'status',r.status)}
if(flag('master')&&!names.length&&!flag('stills')&&flag('only-master')){master();process.exit(0)}
(async()=>{
 const browser=await puppeteer.launch({executablePath:CHROME,headless:'new',args:['--hide-scrollbars','--force-device-scale-factor=1','--font-render-hinting=none']});
 if(val('stills')){ // --stills t1,t2 : PNGs of the named animation(s)
  for(const name of todo){const p=await open(browser,name);for(const t of val('stills').split(',').map(Number)){await p.evaluate(t=>window.render(t),t);await p.screenshot({path:path.join(OUT,'stills',`${name}_${t.toFixed(1)}.png`)})}await p.close()}
  await browser.close();return}
 let q=[...todo];await Promise.all([0,1,2].map(async()=>{while(q.length)await renderOne(browser,q.shift())}));
 await browser.close();
 if(flag('master'))master();
})();
