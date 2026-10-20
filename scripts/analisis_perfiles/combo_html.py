import json, sys
S, NF, NU, OUT = sys.argv[1:5]
data = json.load(open(S + "_combo.json", encoding="utf-8"))
nf = json.load(open(NF, encoding="utf-8")); nu = json.load(open(NU, encoding="utf-8"))
html = """<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Películas y usuarios en 3D</title><style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--card:#f4f4f2}
@media(prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eee;--mut:#9a9a9a;--card:#1f1f1f}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;display:flex;flex-wrap:wrap;height:100vh}
#w{flex:1 1 640px;position:relative;min-height:420px;touch-action:none}canvas{width:100%;height:100%;display:block;cursor:grab}
#side{flex:0 0 340px;padding:14px;overflow:auto;box-sizing:border-box;max-height:100vh}
h1{font-size:17px;margin:0 0 4px}h2{font-size:13px;margin:12px 0 4px}p{color:var(--mut);margin:4px 0 8px;font-size:12px}
.g{display:flex;align-items:center;gap:6px;padding:2px 5px;border-radius:5px;cursor:pointer}.g:hover,.g.on{background:var(--card)}
.g i{width:11px;height:11px;flex:none}.g b{font-weight:600}.g small{color:var(--mut);margin-left:auto}
#tip{position:absolute;pointer-events:none;background:var(--card);padding:4px 8px;border-radius:5px;font-size:12px;display:none;border:1px solid #8884}
button{font:inherit;padding:4px 10px;border-radius:6px;border:1px solid #8886;background:var(--card);color:var(--fg);cursor:pointer;margin:0 4px 6px 0}
button.on{background:var(--fg);color:var(--bg)}label{display:block;margin:2px 0}
</style></head><body><div id="w"><canvas id="c"></canvas><div id="tip"></div></div><div id="side"><h1>Películas y usuarios en 3D</h1>
<p id="info"></p>
<h2>Ejes (espacio de proyección)</h2>
<button id="sp0" class="on">Máxima varianza de las películas</button><button id="sp1">Máxima varianza de los usuarios</button>
<h2>Mostrar</h2>
<label><input type="checkbox" id="sf" checked> Películas (cuadrados, por grupo)</label>
<label><input type="checkbox" id="su" checked> 10.000 usuarios (círculos, por grupo)</label>
<label><input type="checkbox" id="sp" checked> Personas (puntos grandes)</label>
<label><input type="checkbox" id="ef"> Elipsoides de grupos de películas</label>
<label><input type="checkbox" id="eu"> Elipsoides de grupos de usuarios</label>
<div style="margin-top:4px">Tamaño de los semiejes: <button id="s1" class="on">1σ</button><button id="s2">2σ</button></div>
<label><input type="checkbox" id="ec" checked> Relleno translúcido</label>
<div>Opacidad: <input type="range" id="ea" min="0.05" max="0.6" step="0.01" value="0.22" style="vertical-align:middle"></div>
<p>Elipsoide: centro en la media del grupo; semiejes = desviación típica del grupo a lo largo de los 3 ejes principales de su covarianza (se dibujan 3 anillos, uno por plano principal, y los semiejes). Con un grupo resaltado se ve solo el suyo.</p>
<h2>Vista</h2>
<button id="rot">⏸ Pausar giro</button><button id="flat">Plano PC1–PC2</button><button id="rst">Vista 3D</button><button id="lab">Nombres de grupo: sí</button>
<h2>Grupos de usuarios (pulsa para resaltar)</h2><div id="gu"></div>
<h2>Grupos de películas (pulsa para resaltar)</h2><div id="gf"></div>
<p>Arrastra para girar; rueda para acercar.</p></div>
<script>
const D=%DATA%,NF=%NF%,NU=%NU%;
const F=D.films,U=D.users,PE=Object.entries(D.people);
const colF=c=>`hsl(${(c*137.5)%360},60%,55%)`;
const UC=['#e6194b','#2a9d3a','#2f55d4','#f58231','#9b30c0','#111111'];
const cv=document.getElementById('c'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');
let W,H,yaw=.6,pitch=.35,zoom=1,auto=true,labels=true,sp=0,selF=-1,selU=-1,drag=false,sx,sy,last=[];
const $=id=>document.getElementById(id);
const pct=(a,q)=>{a=Float64Array.from(a).sort();return a[Math.floor(a.length*q)]};
const EXT=[0,1].map(s=>{const v=[];for(const p of F)for(let i=0;i<3;i++)v.push(Math.abs(p[s*3+i]));for(const p of U)for(let i=0;i<3;i++)v.push(Math.abs(p[s*3+i]));return pct(v,.995)*1.05});
function proj(x,y,z){const ext=EXT[sp],cy=Math.cos(yaw),sy_=Math.sin(yaw),cp=Math.cos(pitch),sn=Math.sin(pitch);
 let X=x*cy+z*sy_,Z=-x*sy_+z*cy,Y=y*cp-Z*sn;Z=y*sn+Z*cp;const f=2.6/(2.6+Z/ext),s=Math.min(W,H)*.46*zoom/ext*f;return[W/2+X*s,H/2-Y*s,Z,f]}
function fit(){const r=devicePixelRatio||1;W=cv.clientWidth;H=cv.clientHeight;cv.width=W*r;cv.height=H*r;ctx.setTransform(r,0,0,r,0,0)}
const V=()=>sp?D.varU:D.varV;
function info(){const v=V();$('info').textContent=`Espacio: ejes de máxima varianza de ${sp?'los usuarios (200.948)':'las 23.335 películas'}. PC1 ${(v[0]*100).toFixed(1)} %, PC2 ${(v[1]*100).toFixed(1)} %, PC3 ${(v[2]*100).toFixed(1)} % (suman ${((v[0]+v[1]+v[2])*100).toFixed(1)} %). Todo lo demás se proyecta en esos mismos ejes.`}
const cent={f:{},u:{}};
function cents(){for(const [k,arr,ci] of [['f',F,6],['u',U,6]]){const o={};for(const p of arr){const c=p[ci];(o[c]??=[0,0,0,0,0,0,0]);for(let i=0;i<6;i++)o[c][i]+=p[i];o[c][6]++}
 for(const c in o)for(let i=0;i<6;i++)o[c][i]/=o[c][6];cent[k]=o}}
cents();
function eig3(C){let A=C.map(r=>r.slice()),Vv=[[1,0,0],[0,1,0],[0,0,1]];
 for(let it=0;it<40;it++){let p=0,q=1,m=Math.abs(A[0][1]);if(Math.abs(A[0][2])>m){p=0;q=2;m=Math.abs(A[0][2])}if(Math.abs(A[1][2])>m){p=1;q=2;m=Math.abs(A[1][2])}if(m<1e-14)break;
  const th=(A[q][q]-A[p][p])/(2*A[p][q]),t=(th>=0?1:-1)/(Math.abs(th)+Math.sqrt(th*th+1)),c=1/Math.sqrt(t*t+1),s=t*c;
  for(let k=0;k<3;k++){const a=A[k][p],b=A[k][q];A[k][p]=c*a-s*b;A[k][q]=s*a+c*b}
  for(let k=0;k<3;k++){const a=A[p][k],b=A[q][k];A[p][k]=c*a-s*b;A[q][k]=s*a+c*b}
  for(let k=0;k<3;k++){const a=Vv[k][p],b=Vv[k][q];Vv[k][p]=c*a-s*b;Vv[k][q]=s*a+c*b}}
 return{vals:[A[0][0],A[1][1],A[2][2]],vecs:[0,1,2].map(j=>[Vv[0][j],Vv[1][j],Vv[2][j]])}}
const ELL={f:[{},{}],u:[{},{}]};
for(const [key,arr] of [['f',F],['u',U]])for(let s=0;s<2;s++){const o=s*3,g={};for(const p of arr)(g[p[6]]??=[]).push(p);
 for(const c in g){const pts=g[c],n=pts.length,m=[0,0,0];for(const p of pts)for(let i=0;i<3;i++)m[i]+=p[o+i];for(let i=0;i<3;i++)m[i]/=n;
  const C=[[0,0,0],[0,0,0],[0,0,0]];for(const p of pts)for(let i=0;i<3;i++)for(let j=0;j<3;j++)C[i][j]+=(p[o+i]-m[i])*(p[o+j]-m[j]);
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)C[i][j]/=n;const e=eig3(C);ELL[key][s][c]={m,sd:e.vals.map(v=>Math.sqrt(Math.max(v,0))),ax:e.vecs}}}
let sig=1;
function alphaCol(c,a){if(c[0]=='#'){const n=parseInt(c.slice(1),16);return `rgba(${n>>16&255},${n>>8&255},${n&255},${a})`}return c.replace('hsl(','hsla(').replace(')',`,${a})`)}
const SPH=[];for(let i=0;i<=15;i++){const la=-Math.PI/2+i*Math.PI/15;for(let j=0;j<30;j++){const lo=j*Math.PI/15;SPH.push([Math.cos(la)*Math.cos(lo),Math.sin(la),Math.cos(la)*Math.sin(lo)])}}
function hull(P){P.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);const cr=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);const lo=[];for(const p of P){while(lo.length>=2&&cr(lo[lo.length-2],lo[lo.length-1],p)<=0)lo.pop();lo.push(p)}const up=[];for(let i=P.length-1;i>=0;i--){const p=P[i];while(up.length>=2&&cr(up[up.length-2],up[up.length-1],p)<=0)up.pop();up.push(p)}up.pop();lo.pop();return lo.concat(up)}
function fillEll(key,sidx,sel,colf){const E=ELL[key][sidx],al=+$('ea').value,list=[];
 for(const c in E){if(sel>=0&&c!=sel)continue;const{m,sd,ax}=E[c],o=proj(m[0],m[1],m[2]),P=[];
  for(const u of SPH){const a=sig*sd[0]*u[0],b=sig*sd[1]*u[1],d=sig*sd[2]*u[2];const q=proj(m[0]+ax[0][0]*a+ax[1][0]*b+ax[2][0]*d,m[1]+ax[0][1]*a+ax[1][1]*b+ax[2][1]*d,m[2]+ax[0][2]*a+ax[1][2]*b+ax[2][2]*d);P.push([q[0],q[1]])}
  list.push([o[2],o[0],o[1],hull(P),c])}
 list.sort((x,y)=>y[0]-x[0]);
 for(const[z,cx,cy,H,c]of list){let r=0;for(const p of H)r=Math.max(r,Math.hypot(p[0]-cx,p[1]-cy));r=Math.max(r,1);
  const g=ctx.createRadialGradient(cx-.3*r,cy-.35*r,.05*r,cx,cy,r);g.addColorStop(0,alphaCol(colf(c),Math.min(.9,al*1.9)));g.addColorStop(1,alphaCol(colf(c),al*.55));
  ctx.globalAlpha=1;ctx.fillStyle=g;ctx.beginPath();H.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fill()}}
function drawEll(key,sidx,sel,colf){const E=ELL[key][sidx];for(const c in E){if(sel>=0&&c!=sel)continue;const{m,sd,ax}=E[c];ctx.strokeStyle=colf(c);ctx.lineWidth=1.7;ctx.globalAlpha=.9;
 const pt=(a,b,th)=>{const ca=Math.cos(th),sa=Math.sin(th);return proj(m[0]+sig*(ax[a][0]*sd[a]*ca+ax[b][0]*sd[b]*sa),m[1]+sig*(ax[a][1]*sd[a]*ca+ax[b][1]*sd[b]*sa),m[2]+sig*(ax[a][2]*sd[a]*ca+ax[b][2]*sd[b]*sa))};
 for(const[a,b]of[[0,1],[0,2],[1,2]]){ctx.beginPath();for(let k=0;k<=48;k++){const q=pt(a,b,k/48*6.2832);k?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])}ctx.stroke()}
 ctx.globalAlpha=.6;ctx.lineWidth=1;for(let a=0;a<3;a++){const p1=proj(m[0]+sig*ax[a][0]*sd[a],m[1]+sig*ax[a][1]*sd[a],m[2]+sig*ax[a][2]*sd[a]),p2=proj(m[0]-sig*ax[a][0]*sd[a],m[1]-sig*ax[a][1]*sd[a],m[2]-sig*ax[a][2]*sd[a]);ctx.beginPath();ctx.moveTo(p1[0],p1[1]);ctx.lineTo(p2[0],p2[1]);ctx.stroke()}
 const q=proj(m[0],m[1],m[2]);ctx.fillStyle=colf(c);ctx.globalAlpha=1;ctx.beginPath();ctx.arc(q[0],q[1],3.4,0,7);ctx.fill();ctx.lineWidth=1;ctx.strokeStyle='#000';ctx.stroke()}}
const dfac=a=>Math.max(.22,Math.min(1.1,.62-.34*a[0]/EXT[sp]));
function scene(fg,dark,o){const e=EXT[sp],y0=-e,seg=(a,b)=>{ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()};
 const c=[[-e,y0,-e],[e,y0,-e],[e,y0,e],[-e,y0,e]].map(p=>proj(p[0],p[1],p[2]));
 ctx.globalAlpha=dark?.12:.09;ctx.fillStyle=dark?'#8fa3ff':'#3a4a7a';ctx.beginPath();c.forEach((q,i)=>i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));ctx.closePath();ctx.fill();
 ctx.strokeStyle=fg;ctx.lineWidth=1;ctx.globalAlpha=.13;
 for(let i=-4;i<=4;i++){const t=i*e/4;seg(proj(t,y0,-e),proj(t,y0,e));seg(proj(-e,y0,t),proj(e,y0,t))}
 const K=[[-e,-e,-e],[e,-e,-e],[e,e,-e],[-e,e,-e],[-e,-e,e],[e,-e,e],[e,e,e],[-e,e,e]],Q=K.map(p=>proj(p[0],p[1],p[2]));
 for(const [i,j] of [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]){ctx.globalAlpha=(Q[i][2]+Q[j][2])/2>0?.07:.3;seg(Q[i],Q[j])}
 ctx.fillStyle=dark?'#000':'#141a2b';ctx.globalAlpha=dark?.28:.12;
 if($('sf').checked)for(let i=0;i<F.length;i+=6){const p=F[i];if(selF>=0&&p[6]!=selF)continue;const q=proj(p[o],y0,p[o+2]),s=2.2*q[3];ctx.fillRect(q[0]-s,q[1]-s/2,2*s,s)}
 if($('su').checked)for(let i=0;i<U.length;i+=3){const p=U[i];if(selU>=0&&p[6]!=selU)continue;const q=proj(p[o],y0,p[o+2]),s=2.2*q[3];ctx.fillRect(q[0]-s,q[1]-s/2,2*s,s)}
 if($('sp').checked)for(const [n,g] of PE){const q=proj(g[o],y0,g[o+2]);ctx.globalAlpha=.35;ctx.beginPath();ctx.ellipse(q[0],q[1],9*q[3],4*q[3],0,0,7);ctx.fill();const t=proj(g[o],g[o+1],g[o+2]);ctx.globalAlpha=.25;ctx.strokeStyle=fg;seg(q,t)}
 ctx.globalAlpha=1}
function draw(){const dark=matchMedia('(prefers-color-scheme: dark)').matches,gr=ctx.createLinearGradient(0,0,0,H);gr.addColorStop(0,dark?'#242b40':'#ffffff');gr.addColorStop(1,dark?'#0b0c11':'#d9dfec');ctx.globalAlpha=1;ctx.fillStyle=gr;ctx.fillRect(0,0,W,H);
 const fg=getComputedStyle(document.body).color,o=sp*3;scene(fg,dark,o);
 ctx.lineWidth=1;ctx.font='12px system-ui';
 [['PC1',[1,0,0]],['PC2',[0,1,0]],['PC3',[0,0,1]]].forEach(([n,v])=>{const e=EXT[sp],a=proj(-v[0]*e,-v[1]*e,-v[2]*e),b=proj(v[0]*e,v[1]*e,v[2]*e);ctx.globalAlpha=.45;ctx.strokeStyle=fg;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.fillStyle=fg;ctx.fillText(n,b[0]+4,b[1]-4)});
 const arr=[];
 if($('sf').checked)for(let i=0;i<F.length;i++){const p=F[i],q=proj(p[o],p[o+1],p[o+2]);arr.push([q[2],q[0],q[1],q[3],0,i])}
 if($('su').checked)for(let i=0;i<U.length;i++){const p=U[i],q=proj(p[o],p[o+1],p[o+2]);arr.push([q[2],q[0],q[1],q[3],1,i])}
 arr.sort((a,b)=>b[0]-a[0]);last=arr;
 for(const a of arr){if(a[4]==0){const p=F[a[5]],on=selF<0||p[6]==selF;ctx.globalAlpha=(on?.55:.04)*dfac(a);ctx.fillStyle=colF(p[6]);const s=(on?2.4:1.6)*a[3];ctx.fillRect(a[1]-s/2,a[2]-s/2,s,s)}
  else{const p=U[a[5]],on=selU<0||p[6]==selU;ctx.globalAlpha=(on?.9:.05)*dfac(a);ctx.fillStyle=UC[p[6]];ctx.beginPath();ctx.arc(a[1],a[2],(on?2.1:1.4)*a[3],0,7);ctx.fill()}}
 if($('ec').checked){if($('ef').checked)fillEll('f',sp,selF,colF);if($('eu').checked)fillEll('u',sp,selU,c=>UC[c])}
 if($('ef').checked)drawEll('f',sp,selF,colF);if($('eu').checked)drawEll('u',sp,selU,c=>UC[c]);
 ctx.globalAlpha=1;ctx.font='700 12px system-ui';
 if(labels){if($('sf').checked)for(const c in cent.f){if(selF>=0&&c!=selF)continue;const t=cent.f[c],q=proj(t[o],t[o+1],t[o+2]);ctx.fillStyle=fg;ctx.fillText(NF[c],q[0]-30,q[1])}
  if($('su').checked)for(const c in cent.u){if(selU>=0&&c!=selU)continue;const t=cent.u[c],q=proj(t[o],t[o+1],t[o+2]);ctx.fillStyle=UC[c];ctx.font='800 13px system-ui';ctx.fillText('● '+NU[c],q[0]-30,q[1]);ctx.font='700 12px system-ui'}}
 if($('sp').checked){ctx.font='600 13px system-ui';for(const [n,g] of PE){const q=proj(g[o],g[o+1],g[o+2]);ctx.beginPath();ctx.arc(q[0],q[1],8*q[3],0,7);ctx.fillStyle='#fff';ctx.fill();ctx.lineWidth=3;ctx.strokeStyle='#000';ctx.stroke();ctx.fillStyle=fg;ctx.fillText(n,q[0]+12,q[1]+4)}}}
function loop(){if(auto&&!drag)yaw+=.004;draw();requestAnimationFrame(loop)}
cv.onpointerdown=e=>{drag=true;sx=e.clientX;sy=e.clientY;cv.setPointerCapture(e.pointerId);cv.style.cursor='grabbing'};
cv.onpointerup=()=>{drag=false;cv.style.cursor='grab'};
cv.onpointermove=e=>{if(drag){yaw+=(e.clientX-sx)*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-sy)*.008));sx=e.clientX;sy=e.clientY;tip.style.display='none';return}
 const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let bt=null,bd=70,o=sp*3;
 if($('sp').checked)for(const [n,g] of PE){const q=proj(g[o],g[o+1],g[o+2]),dx=q[0]-mx,dy=q[1]-my,d=dx*dx+dy*dy;if(d<300&&d<bd){bd=d;bt=n+' · grupo de usuarios: '+NU[g[6]]}}
 if(!bt)for(let k=last.length-1;k>=0;k--){const a=last[k];if(a[4]==0&&selF>=0&&F[a[5]][6]!=selF)continue;if(a[4]==1&&selU>=0&&U[a[5]][6]!=selU)continue;const dx=a[1]-mx,dy=a[2]-my,d=dx*dx+dy*dy;if(d<bd){bd=d;bt=a[4]==0?F[a[5]][7]+' · '+NF[F[a[5]][6]]:'Usuario · '+NU[U[a[5]][6]]}}
 if(!bt){tip.style.display='none';return}tip.style.display='block';tip.style.left=mx+12+'px';tip.style.top=my+12+'px';tip.textContent=bt};
cv.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(6,zoom*(e.deltaY<0?1.1:.9)))};
function setSp(s){sp=s;$('sp0').classList.toggle('on',s==0);$('sp1').classList.toggle('on',s==1);info()}
$('sp0').onclick=()=>setSp(0);$('sp1').onclick=()=>setSp(1);
$('rot').onclick=e=>{auto=!auto;e.target.textContent=auto?'⏸ Pausar giro':'▶ Girar'};
$('flat').onclick=()=>{yaw=0;pitch=0;auto=false;$('rot').textContent='▶ Girar'};
$('rst').onclick=()=>{yaw=.6;pitch=.35;zoom=1};
$('s1').onclick=()=>{sig=1;$('s1').classList.add('on');$('s2').classList.remove('on')};$('s2').onclick=()=>{sig=2;$('s2').classList.add('on');$('s1').classList.remove('on')};
$('lab').onclick=e=>{labels=!labels;e.target.textContent='Nombres de grupo: '+(labels?'sí':'no')};
const cf=Array(NF.length).fill(0);F.forEach(p=>cf[p[6]]++);
function leyenda(div,nombres,cnt,color,shape,get,set){const el=$(div);for(let c=0;c<nombres.length;c++){const d=document.createElement('div');d.className='g';d.innerHTML=`<i style="background:${color(c)};border-radius:${shape}"></i><b>${nombres[c]}</b><small>${cnt[c]}</small>`;d.onclick=()=>{set(get()==c?-1:c);[...el.children].forEach((e,i)=>e.classList.toggle('on',i==get()))};el.append(d)}}
leyenda('gf',NF,cf,colF,'2px',()=>selF,v=>selF=v);leyenda('gu',NU,D.tamU,c=>UC[c],'50%',()=>selU,v=>selU=v);
addEventListener('resize',fit);fit();info();loop();
</script></body></html>"""
html = html.replace("%DATA%", json.dumps(data, ensure_ascii=False)).replace("%NF%", json.dumps(nf, ensure_ascii=False)).replace("%NU%", json.dumps(nu, ensure_ascii=False))
open(OUT, "w", encoding="utf-8").write(html); print("ok", OUT, len(html) // 1024, "KB")
