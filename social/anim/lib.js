// Shared helpers for the qwenTPU canvas animations. Everything is a pure function of time t (seconds).
const W=1920,H=1080;
const cvs=document.getElementById('c');cvs.width=W;cvs.height=H;
const ctx=cvs.getContext('2d');
const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
const prog=(t,a,b)=>clamp((t-a)/(b-a));
const eo=x=>1-Math.pow(1-x,3);
const eio=x=>x<.5?4*x*x*x:1-Math.pow(-2*x+2,3)/2;
const eob=x=>{const c1=1.70158,c3=c1+1;return 1+c3*Math.pow(x-1,3)+c1*Math.pow(x-1,2)};
const lerp=(a,b,x)=>a+(b-a)*x;
const comma=n=>Math.round(n).toLocaleString('en-US');
const col={bg:'#090c19',panel:'#12172f',panel2:'#181f3f',line:'#2b3566',ink:'#f2f4ff',mut:'#9aa3d4',pu:'#8f7fff',te:'#2fd9bb',am:'#ffb347',pk:'#ff6b8b',gr:'#5ff08a'};
const SANS='"Helvetica Neue",Helvetica,Arial,sans-serif',MONO='Menlo,"SF Mono",Monaco,monospace';
function rgba(hex,a){const n=parseInt(hex.slice(1),16);return `rgba(${n>>16},${(n>>8)&255},${n&255},${a})`}
function mix(h1,h2,x){const a=parseInt(h1.slice(1),16),b=parseInt(h2.slice(1),16);const f=s=>Math.round(lerp((a>>s)&255,(b>>s)&255,x));return `rgb(${f(16)},${f(8)},${f(0)})`}
// background (cached)
const BG=document.createElement('canvas');BG.width=W;BG.height=H;
(()=>{const g=BG.getContext('2d');const gr=g.createRadialGradient(960,420,100,960,540,1200);gr.addColorStop(0,'#131a3a');gr.addColorStop(1,'#070913');g.fillStyle=gr;g.fillRect(0,0,W,H);
 g.fillStyle='rgba(143,127,255,.10)';for(let x=40;x<W;x+=48)for(let y=40;y<H;y+=48){g.beginPath();g.arc(x,y,1.6,0,7);g.fill()}})();
function bg(){ctx.globalAlpha=1;ctx.drawImage(BG,0,0)}
function rr(x,y,w,h,r,fill,stroke,lw=2,alpha=1){ctx.save();ctx.globalAlpha*=alpha;ctx.beginPath();ctx.roundRect(x,y,w,h,r);if(fill){ctx.fillStyle=fill;ctx.fill()}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=lw;ctx.stroke()}ctx.restore()}
function glow(c,b){ctx.shadowColor=c;ctx.shadowBlur=b}
function noglow(){ctx.shadowBlur=0;ctx.shadowColor='transparent'}
function tx(s,x,y,o={}){const {size=40,weight=700,color=col.ink,align='left',mono=false,alpha=1,ls=0,base='alphabetic'}=o;
 ctx.save();ctx.globalAlpha*=alpha;ctx.font=`${weight} ${size}px ${mono?MONO:SANS}`;ctx.fillStyle=color;ctx.textAlign=align;ctx.textBaseline=base;ctx.letterSpacing=ls+'px';
 if(o.glow){glow(o.glow,o.gb||24)}ctx.fillText(s,x,y);ctx.restore()}
function line(x1,y1,x2,y2,color,lw=3,dash,alpha=1){ctx.save();ctx.globalAlpha*=alpha;ctx.strokeStyle=color;ctx.lineWidth=lw;if(dash)ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.restore()}
function dot(x,y,r,color,g=0){ctx.save();if(g)glow(color,g);ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,r,0,7);ctx.fill();ctx.restore()}
function header(idx,label,t){const a=eo(prog(t,0,.6));ctx.save();ctx.globalAlpha=a;
 rr(0,0,W,96,0,'rgba(9,12,25,.92)');line(0,96,W,96,col.line,2);
 rr(48,22,196,52,10,col.pu);tx('qwenTPU',146,60,{size:30,align:'center',color:'#0a0c1c',ls:1.5,weight:800});
 tx(idx+'  '+label,282,62,{size:34,color:col.ink,ls:3});
 tx('A TPU in the weights of Qwen3.5',W-48,60,{size:28,color:col.mut,align:'right',weight:500});ctx.restore()}
function caption(caps,t){for(const [a,b,s] of caps){if(t>=a&&t<=b){const k=Math.min(prog(t,a,a+.35),1-prog(t,b-.3,b));ctx.save();ctx.globalAlpha=k;
  ctx.font=`700 44px ${SANS}`;const w=Math.min(1800,ctx.measureText(s).width+80);rr((W-w)/2,962,w,86,16,'rgba(8,10,22,.94)',col.line,2);
  tx(s,W/2,1020,{size:44,align:'center',color:col.ink});ctx.restore()}}}
function fadeEdges(t,dur){const a=1-prog(t,dur-.35,dur);if(a<1){ctx.save();ctx.globalAlpha=1-a;ctx.fillStyle=col.bg;ctx.fillRect(0,0,W,H);ctx.restore()}}
// register: window.render(t), window.DUR
function register(dur,fn){window.DUR=dur;window.render=t=>{ctx.setTransform(1,0,0,1,0,0);ctx.globalAlpha=1;noglow();bg();fn(t);};window.READY=true}
function pill(s,x,y,color,o={}){const size=o.size||30;ctx.save();ctx.font=`700 ${size}px ${SANS}`;ctx.letterSpacing='2px';const w=ctx.measureText(s).width+56;ctx.restore();
 rr(x,y,w,size+34,14,rgba(color,.13),color,2.5,o.alpha||1);tx(s,x+28,y+size+8,{size,color,ls:2,alpha:o.alpha||1});return w}
