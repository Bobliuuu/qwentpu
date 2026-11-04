// Shared drawing helpers for the v2 demo + clips. Needs light.js (ctx, C, tx, rr, ln, dotc, mixc, ...) and trace_v2.js (window.TRACE2).
// Every number drawn comes from TRACE2 (gen_trace_v2.py: golden.npz + isa.run + the CPU circuit-IR sequencer, all asserted).
const D2=window.TRACE2;
// robust colour mix: accepts #rrggbb or rgb(r,g,b) (the result of an earlier mix)
function mixc(a,b,x){const p=h=>h[0]==='#'?[parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)]:h.match(/[\d.]+/g).slice(0,3).map(Number);const A=p(a),B=p(b);return `rgb(${A.map((v,i)=>Math.round(v+(B[i]-v)*x)).join(',')})`}
const KIND={READ:{c:'#64748b',l:'#e6ebf3'},ACT:{c:C.blue,l:C.bluel},OUT:{c:C.gr,l:C.grl},HALT:{c:C.pu,l:C.pul}};
const wCol=w=>w===0?'#f1f4fa':(w>0?mixc('#e3ecfc','#5b8def',clamp(w/10,.12,1)):mixc('#f0e8fb','#9b6df0',clamp(-w/10,.12,1)));
// 8x8 pixel image, ink dark; reveal 0..1 pops pixels in raster order; cell px size cs
function drawImg(img,x,y,cs,reveal=1,alpha=1){ctx.save();ctx.globalAlpha*=alpha;
 for(let r=0;r<8;r++)for(let c=0;c<8;c++){const i=r*8+c,a=clamp(reveal*70-i,0,1);if(a<=0)continue;const v=img[r][c]/16;
  rr(x+c*cs+1,y+r*cs+1,cs-2,cs-2,3,mixc('#eef1f7','#141a26',Math.pow(v,.85)*a+(1-a)*0),null)}
 ctx.restore()}
// the 15 pooled features laid out as the 4x4 blocks (block 0 is the dropped always-dark corner)
function drawPool(x15,x,y,cs,reveal=1,flash=0){
 for(let b=0;b<16;b++){const r=b>>2,c=b&3,xx=x+c*cs,yy=y+r*cs,a=clamp(reveal*20-b,0,1);if(a<=0)continue;
  if(b===0){rr(xx+1,yy+1,cs-2,cs-2,6,'#f5f7fb','#dfe4ec',1.5);tx('–',xx+cs/2,yy+cs/2+9,{size:26,weight:700,align:'center',color:C.gy});continue}
  const v=x15[b-1];rr(xx+1,yy+1,cs-2,cs-2,6,mixc(mixc('#fff0dd','#f8c58a',v/12),'#ffd9a8',flash),C.or,1.5);
  tx(String(v),xx+cs/2,yy+cs/2+9,{size:26,weight:800,align:'center',color:'#8a4b00',mono:true,alpha:a})}}
function chip(s,x,y,kind,o={}){const k=KIND[kind]||KIND.READ,z=o.size||26;ctx.save();ctx.font=`800 ${z}px ${MONO}`;const w=ctx.measureText(s).width+(o.pad||28);ctx.restore();
 ctx.save();ctx.globalAlpha*=o.alpha??1;rr(x,y,w,z+22,(z+22)/2,o.fill||k.l,k.c,2.5);tx(s,x+w/2,y+z+4,{size:z,weight:800,align:'center',mono:true,color:k.c});ctx.restore();return w}
function tokLabel(t){return t.kind==='READ'?'READ '+t.k:(t.kind==='ACT'?'ACT '+t.k+' = '+t.val:'OUT '+t.val)}
// token j (0..64) of a digit's emitted stream: even = READ, odd = value token, 64 = HALT
function tokAt(d,j){if(j>=64)return{kind:'HALT',k:0,val:0,layer:1};const e=d.toks[j>>1];return(j&1)?{kind:e.kind,k:e.k,val:e.val,layer:e.layer}:{kind:'READ',k:e.k,val:0,layer:e.layer}}
function arrow(x1,y1,x2,y2,c,w=5,dash){ln(x1,y1,x2,y2,c,w,dash);const a=Math.atan2(y2-y1,x2-x1),L=20;ctx.save();ctx.fillStyle=c;ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-L*Math.cos(a-.4),y2-L*Math.sin(a-.4));ctx.lineTo(x2-L*Math.cos(a+.4),y2-L*Math.sin(a+.4));ctx.fill();ctx.restore()}
// position at fraction f (0..1) along a polyline [[x,y],...]
function alongPath(pts,f){let L=0;const seg=[];for(let i=1;i<pts.length;i++){const d=Math.hypot(pts[i][0]-pts[i-1][0],pts[i][1]-pts[i-1][1]);seg.push(d);L+=d}let s=f*L;for(let i=0;i<seg.length;i++){if(s<=seg[i]||i===seg.length-1){const g=clamp(s/seg[i]);return[lerp(pts[i][0],pts[i+1][0],g),lerp(pts[i][1],pts[i+1][1],g)]}s-=seg[i]}return pts[pts.length-1]}
function chipC(s,cx,cy,kind,o={}){const z=o.size||28;ctx.save();ctx.font=`800 ${z}px ${MONO}`;const w=ctx.measureText(s).width+30;ctx.restore();return chip(s,cx-w/2,cy-(z+22)/2,kind,o)}
function box(x,y,w,h,title,sub,col,fill,hl=0){const f=mixc(fill,'#ffffff',0);rr(x,y,w,h,20,fill,col,3+hl*3);if(hl>0){ctx.save();ctx.globalAlpha=hl*.5;rr(x-6,y-6,w+12,h+12,24,null,col,3);ctx.restore()}
 tx(title,x+w/2,y+(sub?h/2-4:h/2+14),{size:46,weight:800,align:'center',color:col});if(sub)tx(sub,x+w/2,y+h/2+38,{size:30,weight:600,align:'center',color:C.mut})}
