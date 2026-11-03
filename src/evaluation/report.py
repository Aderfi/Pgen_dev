"""Static, self-contained HTML report rendered from an evaluation's data files.

Reads only ``metrics.json``, ``predictions.parquet`` and ``curves.json`` — the
report can be regenerated (or replaced by another frontend) without
recomputing anything. Charts are inline SVG and the interactivity (sorting,
filtering, pagination, tooltips) is a few lines of inline JavaScript, so the
file has no external dependencies.
"""

from __future__ import annotations

import json
import logging
from html import escape
from typing import TYPE_CHECKING

import polars as pl

from src.evaluation.schemas import EvaluationReport, MetricCI

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

REPORT_FILE = "report.html"
_HEADLINE = (
    "f1_micro f1_macro f1_weighted f1_samples f2_micro f2_macro f2_weighted f2_samples "
    "f05_micro f05_macro f05_weighted f05_samples precision_micro precision_macro "
    "precision_weighted precision_samples recall_micro recall_macro recall_weighted "
    "recall_samples roc_auc_micro roc_auc_macro pr_auc_micro pr_auc_macro "
    "subset_accuracy hamming_loss jaccard_samples jaccard_macro kappa_mean brier_mean "
    "ece mce"
).split()
_CLASS_COLS = (
    "precision recall specificity f1 f2 f05 roc_auc pr_auc mcc kappa brier "
    "pearson_r pearson_p spearman_rho spearman_p"
).split()

_LOWER_BETTER = {"hamming_loss", "brier_mean", "ece", "mce"}
_FAMILIES = (
    ("F1", ("f1_",)),
    ("F2", ("f2_",)),
    ("F0.5", ("f05_",)),
    ("Precision / recall", ("precision_", "recall_")),
    ("Ranking", ("roc_auc_", "pr_auc_")),
    ("Agreement", ("subset_accuracy", "jaccard_", "kappa_")),
    ("Calibration & error", ("hamming_loss", "brier_", "ece", "mce")),
)
_KPIS = (
    ("f1_macro", "F1 macro"),
    ("f2_macro", "F2 macro"),
    ("pr_auc_macro", "PR-AUC macro"),
    ("roc_auc_macro", "ROC-AUC macro"),
    ("ece", "ECE"),
)
_CLASS_GROUPS = (
    ("", 1),
    ("Counts", 4),
    ("Threshold metrics", 6),
    ("Ranking", 2),
    ("Agreement &amp; error", 3),
    ("Correlation", 4),
)
_CLASS_KIND = {"brier": "lo", "pearson_p": "p", "spearman_p": "p"}
_PRED_GROUPS = {"row_id": "ids", "drugs_cid": "ids"}
_PRED_BARS = {"confidence", "max_tanimoto"}
_LABEL_COLS = {
    "true_labels": "",
    "pred_labels": "",
    "fp_labels": "fp",
    "fn_labels": "fn",
}

_THEME_INIT = (
    "var t;try{t=localStorage.getItem('theme')}catch(e){}"
    "if(!t)t=matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';"
    "document.documentElement.dataset.theme=t"
)

CSS = """
:root{color-scheme:light;
--bg:#fff;--glow:rgba(76,114,176,.13);--card:#fff;--ink:#262626;--mut:#6b6b7b;--line:#dcdde8;--line2:#ecedf3;
--head:#eaeaf2;--grp:#dedfea;--track:#e3e4ee;--chip:#eaeaf2;--seg-on:#fff;--hover:rgba(76,114,176,.08);--navbg:rgba(255,255,255,.86);
--accent:#4c72b0;--ink-a:#4c72b0;--g1:#4c72b0;--g2:#8172b3;--eyebrow:#e3eaf5;
--good-bg:rgba(85,168,104,.18);--good-ink:#2f6d3f;--bad-bg:rgba(196,78,82,.15);--bad-ink:#a23a3e;--bad-c:#c44e52;
--warn-bg:rgba(221,132,82,.2);--warn-ink:#9a4f1f;--info-bg:rgba(76,114,176,.15);--info-ink:#3a5890;
--plot:#eaeaf2;--grid:#fff;--frame:#fff;--cm-off:#eaeaf2;--tipbg:#262626;--tipink:#fff;--pm:#000;--pp:72%;
--sh:0 1px 2px rgba(38,38,38,.05),0 10px 28px -16px rgba(38,38,38,.22);--r:14px;
--p0:#4c72b0;--p1:#dd8452;--p2:#55a868;--p3:#c44e52;--p4:#8172b3;--p5:#937860;--p6:#da8bc3;--p7:#8c8c8c}
[data-theme=dark]{color-scheme:dark;
--bg:#0c1c23;--glow:rgba(31,120,180,.2);--card:#14252c;--ink:#fff;--mut:rgba(255,255,255,.62);--line:rgba(255,255,255,.16);--line2:rgba(255,255,255,.07);
--head:#1a2d35;--grp:#203740;--track:rgba(255,255,255,.13);--chip:rgba(255,255,255,.08);--seg-on:#26404b;--hover:rgba(255,255,255,.055);--navbg:rgba(12,28,35,.86);
--accent:#1f78b4;--ink-a:#a6cee3;--g1:#a6cee3;--g2:#cab2d6;--eyebrow:rgba(31,120,180,.25);
--good-bg:rgba(51,160,44,.24);--good-ink:#b2df8a;--bad-bg:rgba(227,26,28,.24);--bad-ink:#fb9a99;--bad-c:#e31a1c;
--warn-bg:rgba(255,127,0,.22);--warn-ink:#fdbf6f;--info-bg:rgba(31,120,180,.3);--info-ink:#a6cee3;
--plot:rgba(255,255,255,.03);--grid:rgba(255,255,255,.24);--frame:rgba(255,255,255,.24);--cm-off:rgba(255,255,255,.05);--tipbg:#e9eef0;--tipink:#0c1c23;--pm:#fff;--pp:60%;
--sh:0 10px 28px -14px rgba(0,0,0,.75);
--p0:#1f78b4;--p1:#ff7f00;--p2:#33a02c;--p3:#e31a1c;--p4:#cab2d6;--p5:#fdbf6f;--p6:#a6cee3;--p7:#b2df8a}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:70px}
body{margin:0;color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-variant-numeric:tabular-nums;
background:radial-gradient(1100px 460px at 50% -140px,var(--glow),transparent 70%),var(--bg);background-attachment:fixed}
main{max-width:1240px;margin:0 auto;padding:0 24px 90px}
h1,h2,h3,h4{margin:0}h3{font-size:15px;margin:26px 0 10px;text-align:center}code{font:12px ui-monospace,Menlo,monospace;background:var(--chip);padding:1px 6px;border-radius:6px}
.nav{position:sticky;top:0;z-index:50;background:var(--navbg);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}
.nav>div{max-width:1240px;margin:0 auto;padding:9px 24px;display:flex;align-items:center;gap:4px;overflow-x:auto;white-space:nowrap}
.brand{font-weight:700;margin-right:auto;padding-right:16px}#links{display:flex;gap:2px}
.nav a{color:var(--mut);text-decoration:none;padding:5px 12px;border-radius:99px;font-weight:500}.nav a:hover{background:var(--chip);color:var(--ink)}
.nav a.on{background:var(--accent);color:#fff}
#theme{position:relative;flex:none;width:52px;height:28px;margin-left:12px;border:1px solid var(--line);border-radius:99px;background:var(--chip);cursor:pointer;padding:0;font-size:13px;line-height:1}
#theme span{position:absolute;top:50%;transform:translateY(-50%);opacity:.75}#theme .sun{left:8px}#theme .moon{right:8px}
#theme::after{content:"";position:absolute;top:3px;left:3px;width:20px;height:20px;border-radius:50%;background:var(--accent);transition:transform .2s}
[data-theme=dark] #theme::after{transform:translateX(24px)}#theme:focus-visible{outline:2px solid var(--ink-a);outline-offset:2px}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--sh)}
.hero{text-align:center;padding:60px 0 8px}.eyebrow{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-a);
background:var(--eyebrow);padding:4px 12px;border-radius:99px}
.hero h1{font-size:clamp(32px,5vw,52px);letter-spacing:-.03em;margin:14px 0 4px;background:linear-gradient(90deg,var(--g1),var(--g2));-webkit-background-clip:text;background-clip:text;color:transparent}
.scn{font-size:18px;color:var(--mut);margin:0}.hl{margin:22px 0 6px;color:var(--mut)}.sha{color:var(--mut);font-size:12px;margin:6px 0 0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:30px}
.stat{padding:14px 12px}.stat small{display:block;color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.07em}
.stat b{display:block;font-size:24px;letter-spacing:-.02em}.stat span{color:var(--mut);font-size:11.5px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:14px;margin-top:34px;text-align:left}
.kpi{padding:18px 18px 16px;position:relative;overflow:hidden}.kpi::before{content:"";position:absolute;inset:0 0 auto 0;height:4px;background:linear-gradient(90deg,var(--g1),var(--g2))}
.kpi small{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.07em}.kpi .big{display:block;font-size:42px;font-weight:700;letter-spacing:-.03em;line-height:1.15}
.kpi .rb{height:10px;margin:10px 0 6px}.kpi .ci{display:block;font-size:12px}.kpi .chip{margin-top:8px}
.sh{text-align:center;margin:64px 0 24px}.sh h2{font-size:27px;letter-spacing:-.02em}.sh p{color:var(--mut);margin:6px auto 0;max-width:720px}
.sh h2::after{content:"";display:block;width:38px;height:3px;margin:10px auto 0;border-radius:3px;background:linear-gradient(90deg,var(--g1),var(--g2))}
.tools{display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:10px 18px;margin:0 0 14px}.tools .grow{flex:1}
.seg{display:inline-flex;background:var(--chip);border-radius:11px;padding:3px;gap:2px;flex-wrap:wrap}
.seg button{border:0;background:none;padding:6px 13px;border-radius:8px;font:inherit;font-weight:500;color:var(--mut);cursor:pointer}
.seg button.on{background:var(--seg-on);color:var(--ink);box-shadow:0 1px 3px rgba(0,0,0,.18)}.seg em{font-style:normal;opacity:.6;margin-left:4px;font-size:12px}
.chk{display:inline-flex;gap:6px;align-items:center;color:var(--mut);cursor:pointer}.chk input{accent-color:var(--accent)}
input[type=search],select{font:inherit;padding:7px 12px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--ink)}input[type=search]{min-width:280px}
.tw{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--sh)}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:13px}
th,td{padding:8px 12px;border-bottom:1px solid var(--line2);text-align:right;white-space:nowrap}tr:last-child td{border-bottom:0}
th{background:var(--head);font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--mut);cursor:pointer;user-select:none;font-weight:600}
th:hover{color:var(--ink-a)}th[data-asc="1"]::after{content:" ▲";color:var(--ink-a)}th[data-asc="0"]::after{content:" ▼";color:var(--ink-a)}
tr.grp th{cursor:default;text-align:center;color:var(--ink-a);border-bottom:1px solid var(--line);background:var(--grp)}tr.grp th:hover{color:var(--ink-a)}
td:first-child,th:first-child{text-align:left}.t2 td:nth-child(2),.t2 th:nth-child(2){text-align:left}
tbody tr:hover td{box-shadow:inset 0 0 0 100vmax var(--hover)}tr.fs td{border-top:2px solid var(--line)}
.sticky td:first-child,.sticky th:first-child{position:sticky;left:0;z-index:2;background:var(--card);box-shadow:1px 0 0 var(--line)}.sticky th:first-child{background:var(--head)}.sticky tr.grp th:first-child{background:var(--grp)}
td small{display:block;color:var(--mut);font-size:11px;font-weight:400}.ci{color:var(--mut);font-size:11px}
.mv{min-width:130px}#pct td,#pct th{padding:8px 9px}.mvt{display:flex;justify-content:space-between;align-items:baseline;gap:12px}.mvt b{font-size:14.5px}
.rb{position:relative;display:block;height:6px;border-radius:99px;background:var(--track);margin-top:6px}.rb i{position:absolute;display:block}
.rb .band{top:0;bottom:0;border-radius:99px;background:color-mix(in srgb,var(--accent) 45%,transparent)}
.rb .tick{top:-3px;bottom:-3px;width:3px;margin-left:-1.5px;border-radius:2px;background:var(--ink-a)}
.hm{background:color-mix(in srgb,var(--accent) calc(var(--p)*.5%),var(--card))}.hm b{display:block;font-size:13.5px}.noci .ci{display:none}
.hm.best{outline:2px solid var(--ink-a);outline-offset:-2px}.pv{color:var(--mut)}.pv.sig{color:var(--ink);font-weight:600}
.bar{background:linear-gradient(var(--accent),var(--accent)) no-repeat 0 calc(100% - 3px)/var(--w) 4px}
.ov{background:linear-gradient(var(--bad-c),var(--bad-c)) no-repeat 0 calc(100% - 3px)/var(--w) 4px}
.dl{display:flex;flex-wrap:wrap;align-items:center;gap:4px 6px;margin-top:6px}.dl small{display:inline}.base tbody tr:first-child td{background:var(--hover)}
.hide-p .pcol{display:none}th.t,td.t{text-align:left}td.t{max-width:230px;overflow:hidden;text-overflow:ellipsis}
.df{white-space:nowrap}
.badge{display:inline-block;padding:1px 9px;border-radius:99px;font-size:11.5px;font-weight:600;background:var(--chip);color:var(--mut)}
.badge.low,.badge.nw{background:var(--warn-bg);color:var(--warn-ink)}.badge.medium{background:var(--info-bg);color:var(--info-ink)}.badge.high{background:var(--good-bg);color:var(--good-ink)}.badge.bad{background:var(--bad-bg);color:var(--bad-ink)}
.chip{display:inline-block;padding:2px 10px;border-radius:99px;font-size:12px;font-weight:600;background:var(--chip)}.chip.pos{background:var(--good-bg);color:var(--good-ink)}.chip.neg{background:var(--bad-bg);color:var(--bad-ink)}
.lb{white-space:normal;min-width:130px;text-align:left}.pl{font-style:normal;display:inline-block;padding:1px 9px;margin:1px 3px 1px 0;border-radius:99px;font-size:11.5px;font-weight:600;
background:color-mix(in srgb,var(--c) 16%,var(--card));color:color-mix(in srgb,var(--c) var(--pp),var(--pm));border:1px solid color-mix(in srgb,var(--c) 40%,var(--card))}
.pl.nh{opacity:.5;border-style:dashed}th small{display:block;text-transform:none;letter-spacing:0;font-weight:400}.pl.fp::before{content:"+ "}.pl.fn::before{content:"− "}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--c);margin-right:8px}
td.seen[data-s]::before{content:attr(data-s);display:inline-block;padding:1px 9px;border-radius:99px;font-size:11.5px;font-weight:600;background:var(--chip);color:var(--mut)}
td.seen[data-v="0"]::before{background:var(--warn-bg);color:var(--warn-ink)}
.ex{font-weight:700;text-align:center}.ex.ok{color:var(--good-ink)}.ex.bad{color:var(--bad-ink)}.seen{text-align:center}.pp{background:color-mix(in srgb,var(--c) calc(var(--p)*.55%),var(--card))}
.hide-ids .ids,.hide-probs .probs,.hide-seen .seen{display:none}
.c0{--c:var(--p0)}.c1{--c:var(--p1)}.c2{--c:var(--p2)}.c3{--c:var(--p3)}.c4{--c:var(--p4)}.c5{--c:var(--p5)}.c6{--c:var(--p6)}.c7{--c:var(--p7)}.cx{--c:var(--mut)}
.legend{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin:0 0 16px}
.lg{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:99px;padding:5px 13px 5px 9px;font:inherit;font-weight:500;cursor:pointer;display:inline-flex;gap:8px;align-items:center;box-shadow:var(--sh)}
.lg i{width:11px;height:11px;border-radius:50%;background:var(--c)}.lg.off{opacity:.45;text-decoration:line-through}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.chart{margin:0;padding:14px 14px 8px}.chart figcaption{font-weight:600;text-align:center;margin-bottom:4px}.chart svg{width:100%;height:auto;display:block;overflow:visible}
.plot{fill:var(--plot)}.frame{fill:none;stroke:var(--frame)}.gl{stroke:var(--grid);stroke-width:1}.tk{fill:var(--ink);font-size:10.5px;opacity:.85}.ax{fill:var(--ink);font-size:11px;opacity:.9}.diag{stroke:var(--mut);stroke-dasharray:4 4}
.s{transition:opacity .15s}.s polyline{fill:none;stroke:var(--c);stroke-width:2.3;stroke-linejoin:round;stroke-linecap:round}.s circle{fill:var(--c);stroke:var(--card);stroke-width:1.5}
.hv{fill:var(--c);stroke:var(--card);stroke-width:2.5;pointer-events:none;display:none}.dim{opacity:.12}.hid{display:none!important}
.cmgrid{display:flex;flex-wrap:wrap;justify-content:center;gap:14px}.cmc{flex:1 1 175px;max-width:215px;padding:14px}.cmc h4{display:flex;align-items:center;font-size:13px;margin-bottom:10px}
.cmg{display:grid;grid-template-columns:auto 1fr 1fr;gap:4px}.cmg em{font-style:normal;font-size:10.5px;color:var(--mut);align-self:center;text-align:center;padding:0 4px}
.cm{border-radius:8px;text-align:center;padding:10px 4px;line-height:1.15}.cm b{display:block;font-size:17px}.cm small{color:inherit;opacity:.72;font-size:10.5px}
.cm.ok{background:color-mix(in srgb,var(--accent) calc(var(--p)*.8%),var(--cm-off))}.cm.bad{background:color-mix(in srgb,var(--bad-c) calc(var(--p)*.9%),var(--cm-off))}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:18px}.cal{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:16px}
.cal .card{padding:14px 16px;text-align:center}.cal small{color:var(--mut);text-transform:uppercase;letter-spacing:.07em;font-size:11px}.cal b{display:block;font-size:20px;margin:2px 0 6px}.cal s{color:var(--mut);text-decoration:none}
.pager{display:flex;align-items:center;justify-content:center;gap:12px;margin:14px 0 0;color:var(--mut)}.pager button{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:9px;padding:6px 14px;font:inherit;cursor:pointer}
.pager button:disabled{opacity:.4;cursor:default}.pager button:not(:disabled):hover{border-color:var(--ink-a);color:var(--ink-a)}
#tip{position:fixed;z-index:99;display:none;pointer-events:none;background:var(--tipbg);color:var(--tipink);padding:7px 11px;border-radius:9px;font-size:12px;white-space:pre-line;max-width:280px;box-shadow:0 8px 24px rgba(0,0,0,.35)}
#tip::first-line{font-weight:700}
@media(max-width:700px){main{padding:0 12px 60px}input[type=search]{min-width:0;width:100%}.two{grid-template-columns:1fr}}
@media print{.nav,.tools,.pager{display:none}.card,.tw{box-shadow:none}}
"""

JS = """
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const cellVal=c=>c.dataset.v??c.querySelector('[data-v]')?.dataset.v??c.innerText;
function sortTable(th){const t=th.closest('table'),i=th.cellIndex,asc=th.dataset.asc!=='1';
$$('th',th.parentNode).forEach(x=>delete x.dataset.asc);th.dataset.asc=asc?'1':'0';
const rows=[...t.tBodies[0].rows];
rows.sort((a,b)=>{const x=cellVal(a.cells[i]),y=cellVal(b.cells[i]),nx=parseFloat(x),ny=parseFloat(y);
const c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc?c:-c});
rows.forEach(r=>t.tBodies[0].appendChild(r));if(t.id==='preds'){P.rows=rows;pgRender()}}
const root=document.documentElement,sw=$('#theme');
function setTheme(t){root.dataset.theme=t;sw.setAttribute('aria-checked',t==='dark');try{localStorage.setItem('theme',t)}catch(e){}}
sw.onclick=()=>setTheme(root.dataset.theme==='dark'?'light':'dark');sw.setAttribute('aria-checked',root.dataset.theme==='dark');
/* tooltips: any [data-tip] element */
const tip=Object.assign(document.createElement('div'),{id:'tip'});document.body.append(tip);
function showTip(e,t){tip.textContent=t;tip.style.display='block';const w=tip.offsetWidth;
tip.style.left=Math.max(8,Math.min(e.clientX+14,innerWidth-w-8))+'px';tip.style.top=e.clientY+16+'px'}
const hideTip=()=>tip.style.display='none';
document.addEventListener('mousemove',e=>{const t=e.target.closest('[data-tip]');
if(t)showTip(e,t.dataset.tip);else if(!e.target.closest('svg[data-l]'))hideTip()});
/* top nav built from the sections present + scroll-spy */
const secs=$$('main section[id]');
$('#links').innerHTML=secs.map(s=>`<a href="#${s.id}">${s.dataset.nav||s.querySelector('h2')?.textContent||s.id}</a>`).join('');
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)$$('#links a').forEach(a=>a.classList.toggle('on',a.hash==='#'+e.target.id))}),{rootMargin:'-35% 0px -60% 0px'});
secs.forEach(s=>io.observe(s));
/* aggregate: filter by metric family */
function famFilter(b){$$('#fam button').forEach(x=>x.classList.toggle('on',x===b));
$$('#aggt tbody tr').forEach(r=>r.hidden=b.dataset.f!=='all'&&r.dataset.fam!==b.dataset.f)}
/* curves: legend toggles / isolates a class across all charts */
const clsEls=()=>$$('#curves [data-cls]:not(.lg)');
function iso(c){clsEls().forEach(e=>e.classList.toggle('dim',c!==null&&e.dataset.cls!==c))}
function toggleCls(b){b.classList.toggle('off');const off=new Set($$('.lg.off').map(x=>x.dataset.cls));
clsEls().forEach(e=>e.classList.toggle('hid',off.has(e.dataset.cls)))}
$$('svg[data-l]').forEach(svg=>{const L=+svg.dataset.l,T=+svg.dataset.t,W=+svg.dataset.w,H=+svg.dataset.h,dot=$('.hv',svg);
$$('.s',svg).forEach(g=>g._p=g.dataset.pts.split(';').map(s=>s.split(',').map(Number)));
svg.addEventListener('mousemove',e=>{const r=svg.getBoundingClientRect(),k=svg.viewBox.baseVal.width/r.width,
mx=(e.clientX-r.left)*k,my=(e.clientY-r.top)*k;let best=null,bd=900;
$$('.s:not(.hid)',svg).forEach(g=>g._p.forEach(([a,b,n])=>{const px=L+a*W,py=T+(1-b)*H,d=(px-mx)**2+(py-my)**2;
if(d<bd){bd=d;best={g,a,b,n,px,py}}}));
if(best){dot.setAttribute('cx',best.px);dot.setAttribute('cy',best.py);
dot.style.setProperty('--c',getComputedStyle(best.g).getPropertyValue('--c'));dot.style.display='block';
showTip(e,`${best.g.dataset.cls}\\n${svg.dataset.xl}: ${best.a.toFixed(3)}\\n${svg.dataset.yl}: ${best.b.toFixed(3)}`+(best.n!=null?`\\nbin size: ${best.n}`:''))}
else{dot.style.display='none';hideTip()}});
svg.addEventListener('mouseleave',()=>{dot.style.display='none';hideTip()})});
/* predictions: search + filter + pagination (sorting is in sortTable) */
const P={rows:[],page:0,size:25,f:'all',q:''};
function pgRender(){const m=P.rows.filter(r=>{const d=r.dataset,f=P.f;
return(f==='all'||(f==='errors'&&d.exact==='false')||(f==='low'&&d.band==='low')||(f==='novel'&&d.novel==='true'))&&(!P.q||r._q.includes(P.q))});
const n=Math.max(1,Math.ceil(m.length/P.size));P.page=Math.min(P.page,n-1);const a=P.page*P.size;
P.rows.forEach(r=>r.hidden=true);m.slice(a,a+P.size).forEach(r=>r.hidden=false);
$('#pgi').textContent=m.length?`${(a+1).toLocaleString()}–${Math.min(a+P.size,m.length).toLocaleString()} of ${m.length.toLocaleString()}`:'No matches';
$('#prev').disabled=P.page===0;$('#next').disabled=P.page>=n-1}
const pgGo=d=>{P.page+=d;pgRender()};
function pgFilter(b){P.f=b.dataset.f;P.page=0;$$('#pf button').forEach(x=>x.classList.toggle('on',x===b));pgRender()}
const toggleCol=(g,on)=>$('#preds').classList.toggle('hide-'+g,!on);
const pt=$('#preds');if(pt){P.rows=[...pt.tBodies[0].rows];P.rows.forEach(r=>r._q=r.textContent.toLowerCase());pgRender()}
"""


# ── formatting helpers ──────────────────────────────────────────────────────
def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def fmt_ci(ci: MetricCI | None) -> str:
    if ci is None:
        return "—"
    if ci.low is None or ci.high is None:
        return fmt(ci.value)
    return f'{fmt(ci.value)} <span class="ci">[{fmt(ci.low)}–{fmt(ci.high)}]</span>'


def fmt_p(p: float | None) -> str:
    return "—" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def rng(ci: MetricCI | None) -> str:
    if ci is None or ci.low is None or ci.high is None:
        return ""
    return f"{fmt(ci.low)}–{fmt(ci.high)}"


def _rb(ci: MetricCI, scale: float = 1.0) -> str:
    """Forest-plot bar: shaded 95% CI band + a tick at the point estimate."""

    assert ci.value is not None, "ci.value must be set for _rb()"

    def pct(x: float) -> float:
        return max(0.0, min(100.0, 100.0 * x / scale))

    band = ""
    if ci.low is not None and ci.high is not None:
        lo, hi = pct(ci.low), pct(ci.high)
        band = f'<i class="band" style="left:{lo:.1f}%;width:{max(hi - lo, 0.8):.1f}%"></i>'
    return f'<span class="rb">{band}<i class="tick" style="left:{pct(ci.value):.1f}%"></i></span>'


def interval(ci: MetricCI | None, scale: float = 1.0) -> str:
    """Value + CI as text, with the bar underneath."""
    if ci is None or ci.value is None:
        return "—"
    tip = f"{fmt(ci.value)}\n95% CI {rng(ci)}" if rng(ci) else fmt(ci.value)
    return (
        f'<div class="mv" data-tip="{tip}"><div class="mvt"><b>{fmt(ci.value)}</b>'
        f'<span class="ci">{rng(ci)}</span></div>{_rb(ci, scale)}</div>'
    )


def _family(metric: str) -> str:
    for name, prefixes in _FAMILIES:
        if metric.startswith(prefixes):
            return name
    return "Other"


def _ordered_metrics(present: Iterable[str]) -> list[str]:
    """Every metric in ``present``: ``_HEADLINE`` order, unknown ones after, by family."""
    present = set(present)
    order = [name for name, _ in _FAMILIES] + ["Other"]
    keys = [k for k in _HEADLINE if k in present] + sorted(present - set(_HEADLINE))
    return sorted(keys, key=lambda k: order.index(_family(k)))


def _cc(colors: dict[str, int], label: str) -> str:
    return f"c{colors[label] % 8}" if label in colors else "cx"


# ── page skeleton ───────────────────────────────────────────────────────────
def table(
    headers: Sequence[str | tuple[str, str]],
    rows: Iterable[Sequence[str | tuple[str, str]]],
    attrs: str = "",
    groups: Sequence[tuple[str, int]] | None = None,
    sortable: bool = True,
) -> str:
    """A cell (or header) is html, or ``(attributes, html)`` when it needs a class/style."""
    oc = ' onclick="sortTable(this)"' if sortable else ""
    top = ""
    if groups:
        cells = "".join(f'<th colspan="{n}">{g}</th>' for g, n in groups)
        top = f'<tr class="grp">{cells}</tr>'
    head = "".join(
        f"<th{h[0]}{oc}>{h[1]}</th>" if isinstance(h, tuple) else f"<th{oc}>{h}</th>"
        for h in headers
    )

    def td(c: str | tuple[str, str]) -> str:
        return f"<td{c[0]}>{c[1]}</td>" if isinstance(c, tuple) else f"<td>{c}</td>"

    body = "".join("<tr>" + "".join(map(td, r)) + "</tr>" for r in rows)
    return (
        f'<div class="tw"><table {attrs}><thead>{top}<tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def section(id_: str, title: str, body: str, sub: str = "", nav: str = "") -> str:
    sub = f"<p>{sub}</p>" if sub else ""
    nav = f' data-nav="{nav}"' if nav else ""
    return (
        f'<section id="{id_}"{nav}><div class="sh"><h2>{title}</h2>{sub}</div>'
        f"{body}</section>"
    )


def page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{escape(title)}</title><script>{_THEME_INIT}</script>"
        f"<style>{CSS}</style></head><body>"
        f"<nav class='nav'><div><span class='brand'>{escape(title)}</span>"
        "<span id='links'></span>"
        "<button id='theme' role='switch' aria-label='Dark theme' title='Switch light / dark theme'>"
        "<span class='sun'>☀</span><span class='moon'>☾</span></button></div></nav>"
        f"<main>{body}</main><script>{JS}</script></body></html>"
    )


# ── header + KPI cards ──────────────────────────────────────────────────────
def stat(label: str, value: str, sub: str = "") -> str:
    sub = f"<span>{sub}</span>" if sub else ""
    return f'<div class="stat card"><small>{label}</small><b>{value}</b>{sub}</div>'


def delta_chip(d: float | None, lower_better: bool = False, tip: str = "") -> str:
    """Signed difference as a chip: green when it favours the model, red otherwise."""
    if d is None:
        return "—"
    kind = "" if d == 0 else "pos" if (d < 0 if lower_better else d > 0) else "neg"
    tip = f' data-tip="{escape(tip)}"' if tip else ""
    return f'<span class="chip {kind}"{tip}>{d:+.3f}</span>'


def _best_baseline(r: EvaluationReport, k: str):
    cands = [
        b
        for b in r.baselines
        if k in b.vs_model and b.aggregate.get(k) and b.aggregate[k].value is not None
    ]
    pick = min if k in _LOWER_BETTER else max
    return pick(cands, key=lambda b: b.aggregate[k].value, default=None)


def _kpis(r: EvaluationReport) -> str:
    cards = []
    for k, label in _KPIS:
        ci = r.aggregate.get(k)
        if ci is None or ci.value is None:
            continue
        lower = k in _LOWER_BETTER
        base = _best_baseline(r, k)
        delta = ""
        if base is not None and base.vs_model[k].diff.value is not None:
            d = base.vs_model[k].diff.value
            ahead = d < 0 if lower else d > 0
            delta = (
                f'<span class="chip {"pos" if ahead else "neg"}" '
                f'data-tip="model − {escape(base.name)} (best baseline)">'
                f"{d:+.3f} vs {escape(base.name)}</span>"
            )
        hint = " ↓" if lower else ""
        cards.append(
            f'<div class="kpi card"><small>{label}{hint}</small>'
            f'<span class="big">{fmt(ci.value)}</span>'
            f"{_rb(ci, 0.5 if lower else 1.0)}"
            f'<span class="ci">95% CI {rng(ci) or "—"}</span>{delta}</div>'
        )
    return f'<div class="kpis">{"".join(cards)}</div>'


def _header(r: EvaluationReport, colors: dict[str, int]) -> str:
    headline = set(r.headline_labels)
    pills = "".join(
        f'<span class="pl {_cc(colors, lbl)}">{escape(lbl)}</span>'
        if lbl in headline
        else f'<span class="pl nh {_cc(colors, lbl)}" data-tip="fewer than 10 positives'
        f'\nexcluded from headline metrics">{escape(lbl)}</span>'
        for lbl in dict.fromkeys([*r.labels, *r.headline_labels])
    )
    stats = (
        stat("Test rows", f"{r.n_test:,}", "out-of-fold")
        + stat("Never tested", f"{r.rows_never_tested:,}")
        + stat("Dropped", f"{r.rows_dropped:,}", "strict cold, all folds")
        + stat("Folds", ", ".join(map(str, r.folds_evaluated)), f"of k={r.k}")
        + stat("Seed", str(r.seed))
        + stat("Bootstrap", f"{r.n_bootstrap:,}", "resamples · 95% CI")
    )
    return (
        '<section id="overview" class="hero" data-nav="Overview">'
        '<span class="eyebrow">Model evaluation</span>'
        f"<h1>{escape(r.model)}</h1><p class='scn'>{escape(str(r.scenario))}</p>"
        f'<div class="stats">{stats}</div>'
        f'<p class="hl">Evaluated classes · headline = ≥10 positives &nbsp;{pills}</p>'
        f'<p class="sha">data <code>{r.data_sha256[:12]}</code> · split '
        f"<code>{r.split_sha256[:12]}</code> · {r.created_at:%Y-%m-%d %H:%M} UTC</p>"
        f"{_kpis(r)}</section>"
    )


# ── aggregate ───────────────────────────────────────────────────────────────
def _calibration(r: EvaluationReport) -> str:
    cal, cards = r.calibration, []
    for label, key in (("ECE", "ece"), ("MCE", "mce"), ("Brier", "brier")):
        b, a = cal[f"{key}_before"], cal[f"{key}_after"]
        d = "" if b is None or a is None else a - b
        chip = (
            ""
            if d == ""
            else f'<span class="chip {"pos" if d < 0 else "neg"}">{d:+.3f}</span>'
        )
        cards.append(
            f'<div class="card"><small>{label} · temperature scaling</small>'
            f"<b>{fmt(b)} <s>→</s> {fmt(a)}</b>{chip}</div>"
        )
    return f'<div class="cal">{"".join(cards)}</div>'


def _aggregate(r: EvaluationReport) -> str:
    keys = _ordered_metrics(r.aggregate)
    fams = list(dict.fromkeys(_family(k) for k in keys))
    chips = (
        '<button class="on" data-f="all" onclick="famFilter(this)">All</button>'
        + "".join(
            f'<button data-f="{f}" onclick="famFilter(this)">{f}</button>' for f in fams
        )
    )
    rows, seen = [], set()
    for k in keys:
        fam, lower = _family(k), k in _LOWER_BETTER
        sc = 0.5 if lower else 1.0
        bf = r.between_folds[k] if k in r.between_folds else None
        if bf is None:
            folds = "—"
        else:
            folds = fmt(bf.mean) if bf.sd is None else f"{fmt(bf.mean)} ± {fmt(bf.sd)}"
        hint = "<small>↓ lower is better</small>" if lower else ""
        first = "" if fam in seen else ' class="fs"'
        seen.add(fam)
        rows.append(
            f'<tr data-fam="{fam}"{first}><td><b>{k}</b>{hint}</td>'
            f"<td>{interval(r.aggregate.get(k), sc)}</td>"
            f"<td>{interval(r.aggregate_at_05.get(k), sc)}</td>"
            f"<td>{interval(r.aggregate_all_classes.get(k), sc)}</td>"
            f'<td class="ci">{folds}</td></tr>'
        )
    head = "".join(
        f"<th>{h}</th>"
        for h in (
            "metric",
            "headline (val thresholds)",
            "headline @0.5",
            "all classes",
            "between folds (mean ± sd)",
        )
    )
    body = (
        f'<div class="tools"><div class="seg" id="fam">{chips}</div></div>'
        f'<div class="tw"><table id="aggt"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>{_calibration(r)}"
    )
    return section(
        "aggregate",
        "Aggregate metrics",
        body,
        "Each bar shows the 95% bootstrap interval; the tick marks the point estimate.",
        "Aggregate",
    )


# ── per fold ────────────────────────────────────────────────────────────────
def _fold_td(k: str, v: float | None, best: float | None):
    if v is None:
        return "—"
    x = max(0.0, 1 - v / 0.5) if k in _LOWER_BETTER else max(0.0, min(1.0, v))
    star = " best" if best == v else ""
    return (
        f' class="hm{star}" style="--p:{x * 100:.0f}" data-v="{v}"',
        f"<b>{fmt(v)}</b>",
    )


def _per_fold(r: EvaluationReport) -> str:
    if not r.per_fold:
        return ""
    folds = sorted(r.per_fold, key=lambda f: f.fold)
    keys = _ordered_metrics(k for f in folds for k in f.aggregate)
    rows, seen = [], set()
    for k in keys:
        vals = [f.aggregate.get(k) for f in folds]
        known = [v for v in vals if v is not None]
        lower = k in _LOWER_BETTER
        best = (min if lower else max)(known) if len(known) > 1 else None
        spread = fmt(max(known) - min(known)) if len(known) > 1 else "—"
        fam = _family(k)
        first = "" if fam in seen else ' class="fs"'
        seen.add(fam)
        hint = "<small>↓ lower is better</small>" if lower else ""
        cells = "".join(
            f"<td{c[0]}>{c[1]}</td>" if isinstance(c, tuple) else f"<td>{c}</td>"
            for c in (_fold_td(k, v, best) for v in vals)
        )
        rows.append(
            f"<tr{first}><td><b>{k}</b>{hint}</td>{cells}"
            f'<td class="ci">{spread}</td></tr>'
        )
    head = "".join(
        f'<th onclick="sortTable(this)">{h}</th>'
        for h in (
            "metric",
            *(f"fold {f.fold}<small>n={f.n_test:,}</small>" for f in folds),
            "spread (max − min)",
        )
    )
    body = (
        f'<div class="tw"><table id="foldt" class="sticky"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
    return section(
        "folds",
        "Per fold",
        body,
        "Headline metrics on each fold's test set (thresholds fitted on that fold's val). "
        "Outlined cell = best fold for that metric.",
        "Folds",
    )


# ── per class ───────────────────────────────────────────────────────────────
def _heat_td(k: str, ci: MetricCI | None, best: dict[str, float]):
    if ci is None or ci.value is None:
        return "—"
    kind, v = _CLASS_KIND.get(k, "hi"), ci.value
    if kind == "p":
        return (f' class="pv pcol" data-v="{v}"', fmt_p(v))
    x = max(0.0, 1 - v / 0.25) if kind == "lo" else max(0.0, min(1.0, v))
    star = " best" if best.get(k) == v else ""
    sub = f'<small class="ci">{rng(ci)}</small>' if rng(ci) else ""
    return (
        f' class="hm{star}" style="--p:{x * 100:.0f}" data-v="{v}"',
        f"<b>{fmt(v)}</b>{sub}",
    )


def _per_class(r: EvaluationReport, colors: dict[str, int]) -> str:
    reliable = [c for c in r.per_class if c.reliable]
    best: dict[str, float] = {}
    for k in _CLASS_COLS:
        kind = _CLASS_KIND.get(k, "hi")
        vals = [
            getattr(c, k).value
            for c in reliable
            if getattr(c, k) is not None and getattr(c, k).value is not None
        ]
        if kind != "p" and len(vals) > 1:
            best[k] = min(vals) if kind == "lo" else max(vals)
    top = max((c.support for c in r.per_class), default=1) or 1
    rows = []
    for c in r.per_class:
        flag = (
            ""
            if c.reliable
            else ' <span class="badge low" data-tip="fewer than 10 positives">n&lt;10</span>'
        )
        rows.append(
            [
                f'<span class="dot {_cc(colors, c.label)}"></span><b>{escape(c.label)}</b>{flag}',
                (
                    f' class="bar" style="--w:{100 * c.support / top:.0f}%"',
                    str(c.support),
                ),
                fmt(c.prevalence),
                fmt(c.threshold),
                (
                    f' data-tip="TP {c.tp}\nFP {c.fp}\nFN {c.fn}\nTN {c.tn}"',
                    f"{c.tp}/{c.fp}/{c.fn}/{c.tn}",
                ),
                *(_heat_td(k, getattr(c, k), best) for k in _CLASS_COLS),
            ]
        )
    heads = [
        "class",
        "support",
        "prevalence",
        "threshold",
        "TP/FP/FN/TN",
        *((' class="pcol"', k) if k.endswith("_p") else k for k in _CLASS_COLS),
    ]
    tools = (
        '<div class="tools"><label class="chk"><input type="checkbox" checked '
        "onchange=\"$('#pct').classList.toggle('noci',!this.checked)\">Confidence intervals</label>"
        '<label class="chk"><input type="checkbox" checked '
        "onchange=\"$('#pct').classList.toggle('hide-p',!this.checked);"
        "$('#pct .grp th:last-child').colSpan=this.checked?4:2\">p-values</label></div>"
    )
    return section(
        "classes",
        "Per class",
        tools
        + table(heads, rows, attrs='id="pct" class="sticky"', groups=_CLASS_GROUPS),
        "Stronger colour = better. Outlined cell = best class for that metric. pearson_r is the "
        "point-biserial correlation between label and probability; MCC is the phi "
        "coefficient of the thresholded predictions.",
        "Per class",
    )


# ── curves (inline SVG) ─────────────────────────────────────────────────────
_W, _H, _ML, _MR, _MT, _MB = 400, 320, 46, 14, 14, 42


def _thin(xs: Sequence[float], ys: Sequence[float], n: int = 600):
    if len(xs) <= n:
        return list(xs), list(ys)
    idx = sorted({round(i * (len(xs) - 1) / (n - 1)) for i in range(n)})
    return [xs[i] for i in idx], [ys[i] for i in idx]


def _chart(title, xlabel, ylabel, series, colors, diagonal=False, markers=False) -> str:
    pw, ph = _W - _ML - _MR, _H - _MT - _MB

    def X(v: float) -> float:
        return _ML + v * pw

    def Y(v: float) -> float:
        return _MT + (1 - v) * ph

    g = [f'<rect class="plot" x="{_ML}" y="{_MT}" width="{pw}" height="{ph}"/>']
    for t in (0, 0.25, 0.5, 0.75, 1):
        g.append(
            f'<line class="gl" x1="{_ML}" x2="{_W - _MR}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
            f'<line class="gl" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{_MT}" y2="{_H - _MB}"/>'
            f'<text class="tk" x="{_ML - 8}" y="{Y(t) + 3.5:.1f}" text-anchor="end">{t:g}</text>'
            f'<text class="tk" x="{X(t):.1f}" y="{_H - _MB + 16}" text-anchor="middle">{t:g}</text>'
        )
    g.append(f'<rect class="frame" x="{_ML}" y="{_MT}" width="{pw}" height="{ph}"/>')

    if diagonal:
        g.append(
            f'<line class="diag" x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}"/>'
        )
    for label, (xs, ys, ns) in series.items():
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys, strict=True))
        raw = ";".join(
            f"{x:.4f},{y:.4f}" + (f",{ns[i]}" if ns else "")
            for i, (x, y) in enumerate(zip(xs, ys, strict=True))
        )
        dots = (
            "".join(
                f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.6"/>'
                for x, y in zip(xs, ys, strict=True)
            )
            if markers
            else ""
        )
        g.append(
            f'<g class="s {_cc(colors, label)}" data-cls="{escape(label)}" data-pts="{raw}">'
            f'<polyline points="{pts}"/>{dots}</g>'
        )
    return (
        f'<figure class="chart card"><figcaption>{title}</figcaption>'
        f'<svg viewBox="0 0 {_W} {_H}" data-l="{_ML}" data-t="{_MT}" data-w="{pw}" '
        f'data-h="{ph}" data-xl="{xlabel}" data-yl="{ylabel}">{"".join(g)}'
        f'<text class="ax" x="{_ML + pw / 2}" y="{_H - 6}" text-anchor="middle">{xlabel}</text>'
        f'<text class="ax" transform="translate(12 {_MT + ph / 2}) rotate(-90)" '
        f'text-anchor="middle">{ylabel}</text><circle class="hv" r="5.5"/></svg></figure>'
    )


def _curve_series(curves: dict, x: str, y: str) -> dict:
    return {lbl: (*_thin(c[x], c[y]), None) for lbl, c in curves.items() if c[x]}


def _reliability_series(curves: dict) -> dict:
    out = {}
    for lbl, c in curves.items():
        pts = [
            (m, f, n)
            for m, f, n in zip(
                c["rel_mean_pred"], c["rel_frac_pos"], c["rel_count"], strict=True
            )
            if n
        ]
        if pts:
            xs, ys, ns = zip(*pts, strict=True)
            out[lbl] = (list(xs), list(ys), list(ns))
    return out


def _legend(colors: dict[str, int]) -> str:
    btns = "".join(
        f'<button class="lg {_cc(colors, lbl)}" data-cls="{escape(lbl)}" onclick="toggleCls(this)" '
        f'onmouseenter="iso(this.dataset.cls)" onmouseleave="iso(null)"><i></i>{escape(lbl)}</button>'
        for lbl in colors
    )
    return f'<div class="legend">{btns}</div>'


def _confusion(r: EvaluationReport, colors: dict[str, int]) -> str:
    cards = []
    for c in r.per_class:
        cells = []
        for pair, who in (
            ((("tn", c.tn), ("fp", c.fp)), "negatives"),
            ((("fn", c.fn), ("tp", c.tp)), "positives"),
        ):
            tot = sum(v for _, v in pair) or 1
            for name, v in pair:
                rate = v / tot
                kind = "ok" if name in ("tn", "tp") else "bad"
                cells.append(
                    f'<div class="cm {kind}" style="--p:{rate * 100:.0f}" '
                    f'data-tip="{name.upper()}: {v}\n{rate:.1%} of actual {who}">'
                    f"<b>{v}</b><small>{rate:.0%}</small></div>"
                )
        tn, fp, fn, tp = cells
        cards.append(
            f'<div class="cmc card {_cc(colors, c.label)}" data-cls="{escape(c.label)}">'
            f'<h4><span class="dot"></span>{escape(c.label)}</h4><div class="cmg">'
            f"<span></span><em>pred 0</em><em>pred 1</em><em>true 0</em>{tn}{fp}<em>true 1</em>{fn}{tp}"
            "</div></div>"
        )
    return f'<div class="cmgrid">{"".join(cards)}</div>'


def _plots(r: EvaluationReport, curves: dict, colors: dict[str, int]) -> str:
    charts = (
        _chart("Precision–recall", "recall", "precision", _curve_series(curves, "pr_recall", "pr_precision"), colors)
        + _chart("ROC", "false positive rate", "true positive rate", _curve_series(curves, "roc_fpr", "roc_tpr"), colors, diagonal=True)
        + _chart("Reliability (calibrated)", "mean predicted", "fraction positive", _reliability_series(curves), colors, diagonal=True, markers=True)
    )  # fmt: skip
    return section(
        "curves",
        "Curves &amp; confusion",
        f'{_legend(colors)}<div class="charts">{charts}</div>'
        f"<h3>Confusion matrices</h3>{_confusion(r, colors)}",
        "Click a class to hide it · hover to isolate it · hover a curve to read exact values. "
        "Confusion cells show the share of each true class.",
        "Curves",
    )


# ── leakage / baselines ─────────────────────────────────────────────────────
def _strata_table(strata) -> str:
    top = max((s.n for s in strata), default=1) or 1
    rows = [
        [
            escape(s.flag),
            escape(s.value),
            (f' class="bar" style="--w:{100 * s.n / top:.0f}%"', str(s.n)),
            interval(s.f1_macro),
        ]
        for s in strata
    ]
    return table(
        ["flag", "value", "n", "F1 macro (headline)"], rows, attrs='class="t2"'
    )


def _ov_cell(o, f: str):
    pct = getattr(o, f + "_pct")
    return (
        f' class="ov" style="--w:{min(pct, 100):.0f}%"',
        f"{getattr(o, f)} ({pct:.1f}%)",
    )


def _leakage(r: EvaluationReport) -> str:
    fields = [
        "exact_duplicates",
        "shared_pmids",
        "shared_pairs",
        "shared_drugs",
        "shared_variants",
        "shared_genes",
    ]
    rows = []
    for fold, o in zip(r.folds_evaluated, r.overlap, strict=False):
        ok = '<span class="badge high">none</span>'
        bad = (
            f'<span class="badge bad">{", ".join(o.violations)}</span>'
            if o.violations
            else ok
        )
        rows.append(
            [str(fold), f"{o.n_test:,}", *(_ov_cell(o, f) for f in fields), bad]
        )
    body = (
        '<div class="two"><div><h3>By novelty (seen in the fold\'s train?)</h3>'
        f"{_strata_table(r.novelty)}</div>"
        f"<div><h3>By max Tanimoto similarity to a train drug</h3>{_strata_table(r.tanimoto)}</div></div>"
        "<h3>Train/test overlap per fold</h3>"
        + table(
            ["fold", "n test", *(f.replace("_", " ") for f in fields), "violations"],
            rows,
        )
    )
    return section(
        "leakage", "Leakage &amp; novelty", body,
        "How performance changes on unseen drugs / variants, and how much train and test overlap.",
        "Leakage",
    )  # fmt: skip


def _diff(v) -> str:
    d = v.diff
    if d is None or d.value is None:
        return "—"
    sig = v.p_value is not None and v.p_value < 0.05
    ci = f'<small class="ci">{rng(d)}</small>' if rng(d) else ""
    verdict = "significant" if sig else "not significant"
    return (
        f'<div class="df"><span class="chip {"pos" if d.value > 0 else "neg"}">{d.value:+.3f}</span> '
        f'<span class="pv{" sig" if sig else ""}" data-tip="{verdict} at 0.05">p={fmt(v.p_value, 4)}</span></div>{ci}'
    )


def _baselines(r: EvaluationReport) -> str:
    keys = ("f1_macro", "f2_macro", "pr_auc_macro")
    rows = [["<b>model</b>", *(interval(r.aggregate.get(k)) for k in keys), "", "", ""]]
    for b in r.baselines:
        rows.append(
            [
                escape(b.name),
                *(interval(b.aggregate.get(k)) for k in keys),
                *(_diff(b.vs_model[k]) for k in keys),
            ]
        )
    return section(
        "baselines",
        "Baselines",
        table(["", *keys, *(f"Δ {k}" for k in keys)], rows, attrs='class="base"'),
        "Δ = model − baseline, with bootstrap 95% CI and p-value. Green = model ahead.",
        "Baselines",
    )


# ── individual predictions ──────────────────────────────────────────────────
def _col_group(c: str) -> str:
    if c in _PRED_GROUPS:
        return _PRED_GROUPS[c]
    return "probs" if c.startswith("p_") else "seen" if c.startswith("seen_") else ""


def _pills(v, colors: dict[str, int], kind: str = "") -> str:
    parts = v if isinstance(v, list) else [p for p in str(v).split("|") if p]
    return "".join(
        f'<i class="pl {kind} {_cc(colors, p)}">{escape(p)}</i>' for p in parts
    )


def _pred_td(c: str, v, colors: dict[str, int]) -> str:  # noqa: PLR0911  (one return per column kind)
    g = _col_group(c)
    if v is None:
        return f'<td class="{g}">—</td>'
    if c in _LABEL_COLS:
        return f'<td class="lb" data-v="{escape(str(v))}">{_pills(v, colors, _LABEL_COLS[c])}</td>'
    if isinstance(v, bool):
        if c == "exact_match":
            return f'<td class="ex {"ok" if v else "bad"}" data-v="{int(v)}">{"✓" if v else "✗"}</td>'
        if c.startswith(
            "seen_"
        ):  # label drawn by CSS (td.seen::before) to keep the file small
            return f'<td class="seen" data-v="{int(v)}" data-s="{"seen" if v else "new"}"></td>'
    if c == "confidence_band":
        return f'<td><span class="badge {escape(str(v))}">{escape(str(v))}</span></td>'
    if c in _PRED_BARS and isinstance(v, float):
        return f'<td class="bar" style="--w:{max(0.0, min(v, 1.0)) * 100:.0f}%">{fmt(v)}</td>'
    if c.startswith("p_") and isinstance(v, float):
        return f'<td class="pp probs {_cc(colors, c[2:])}" style="--p:{v * 100:.0f}">{fmt(v)}</td>'
    if isinstance(v, str):
        tip = f' data-tip="{escape(v)}"' if len(v) > 28 else ""
        return f'<td class="t {g}"{tip}>{escape(v)}</td>'
    text = fmt(v) if isinstance(v, float) else escape(str(v))
    return f'<td class="{g}">{text}</td>'


def _predictions(preds: pl.DataFrame, colors: dict[str, int]) -> str:
    cols = preds.columns
    left = {
        c for c, t in preds.schema.items() if t == pl.String and c != "confidence_band"
    }
    head = "".join(
        f'<th class="{_col_group(c)}{" t" if c in left else ""}" onclick="sortTable(this)">{escape(c)}</th>'
        for c in cols
    )
    body, n_err, n_low, n_novel = [], 0, 0, 0
    for row in preds.iter_rows(named=True):
        novel = any(row.get(c) is False for c in ("seen_drug", "seen_variant"))
        err = row.get("exact_match") is False
        low = row.get("confidence_band") == "low"
        n_err, n_low, n_novel = n_err + err, n_low + low, n_novel + novel
        body.append(
            f"<tr data-exact='{str(row.get('exact_match')).lower()}' "
            f"data-band='{row.get('confidence_band')}' data-novel='{str(novel).lower()}'>"
            + "".join(_pred_td(c, row[c], colors) for c in cols)
            + "</tr>"
        )
    seg = "".join(
        f'<button class="{"on" if f == "all" else ""}" data-f="{f}" onclick="pgFilter(this)">'
        f"{t}<em>{k:,}</em></button>"
        for f, t, k in (
            ("all", "All", len(body)),
            ("errors", "Errors", n_err),
            ("low", "Low confidence", n_low),
            ("novel", "Novel drug / variant", n_novel),
        )
    )
    tools = (
        '<div class="tools"><input type="search" placeholder="Search drug, gene, genotype, label…" '
        'oninput="P.q=this.value.trim().toLowerCase();P.page=0;pgRender()">'
        f'<div class="seg" id="pf">{seg}</div></div>'
        '<div class="tools"><label class="chk"><input type="checkbox" '
        "onchange=\"toggleCol('ids',this.checked)\">IDs</label>"
        '<label class="chk"><input type="checkbox" checked onchange="toggleCol(\'probs\',this.checked)">'
        "Probabilities</label>"
        '<label class="chk"><input type="checkbox" checked onchange="toggleCol(\'seen\',this.checked)">'
        "Seen flags</label>"
        '<label class="chk">Rows <select onchange="P.size=+this.value;P.page=0;pgRender()">'
        "<option>10</option><option selected>25</option><option>50</option><option>100</option>"
        "</select></label></div>"
    )
    pager = (
        '<div class="pager"><button id="prev" onclick="pgGo(-1)">‹ Prev</button>'
        '<span id="pgi"></span><button id="next" onclick="pgGo(1)">Next ›</button></div>'
    )
    return section(
        "predictions",
        "Individual predictions",
        f'{tools}<div class="tw"><table id="preds" class="hide-ids"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>{pager}",
        "Click a header to sort the whole set · also in predictions.csv",
        "Predictions",
    )


# ── entry point ─────────────────────────────────────────────────────────────
def _colors(report: EvaluationReport, curves: dict) -> dict[str, int]:
    labels = [c.label for c in report.per_class]
    labels += [lbl for lbl in curves if lbl not in labels]
    return {lbl: i for i, lbl in enumerate(labels)}


def render(eval_dir: Path) -> Path:
    """Write ``report.html`` into ``eval_dir`` from its data files."""
    report = EvaluationReport.model_validate_json(
        (eval_dir / "metrics.json").read_text()
    )
    curves = json.loads((eval_dir / "curves.json").read_text())
    preds = pl.read_parquet(eval_dir / "predictions.parquet")
    colors = _colors(report, curves)
    body = (
        _header(report, colors)
        + _aggregate(report)
        + _per_fold(report)
        + _per_class(report, colors)
        + _plots(report, curves, colors)
        + _leakage(report)
        + _baselines(report)
        + _predictions(preds, colors)
    )
    out = eval_dir / REPORT_FILE
    out.write_text(page(f"{report.model} · {report.scenario}", body))
    logger.info("Report written to %s", out)
    return out


__all__ = [
    "REPORT_FILE",
    "delta_chip",
    "fmt",
    "fmt_ci",
    "interval",
    "page",
    "render",
    "section",
    "stat",
    "table",
]
