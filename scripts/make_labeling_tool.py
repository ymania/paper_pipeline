#!/usr/bin/env python3
"""
生成浏览器标注页面 — 看图点标签，1秒1个
输出: output/ground_truth/labeling_tool.html
"""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
TEMPLATE = PROJECT / 'output' / 'ground_truth' / 'template.json'

with open(TEMPLATE, 'r') as f:
    data = json.load(f)

samples_json = json.dumps(data['samples'], ensure_ascii=False)
classes_json = json.dumps(data['classes'], ensure_ascii=False)

html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>语义标注工具 — Ground Truth Labeling</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: system-ui; background: #1a1a2e; color: #eee; }}
  .top {{ display:flex; padding:10px 20px; background:#16213e; align-items:center; gap:15px; }}
  .top h1 {{ font-size:18px; }}
  .progress {{ color:#0f3460; background:#0f3460; height:4px; }}
  .progress-bar {{ height:100%; background:#e94560; transition:width 0.2s; }}
  .main {{ display:flex; flex-direction:column; align-items:center; padding:20px; }}
  .patch {{ width:384px; height:384px; object-fit:contain; border:2px solid #333; border-radius:8px; background:#000; image-rendering:pixelated; }}
  .info {{ margin:10px 0; font-size:14px; color:#999; }}
  .info span {{ color:#e94560; font-weight:bold; }}
  .buttons {{ display:flex; flex-wrap:wrap; gap:8px; max-width:500px; justify-content:center; margin:15px 0; }}
  .btn {{ padding:10px 20px; border:none; border-radius:6px; font-size:15px; cursor:pointer; color:#fff; transition:0.1s; }}
  .btn:hover {{ transform:scale(1.05); opacity:0.9; }}
  .btn-wall {{ background:#e94560; }}
  .btn-building {{ background:#c0392b; }}
  .btn-window {{ background:#2980b9; }}
  .btn-door {{ background:#8e44ad; }}
  .btn-floor {{ background:#7f8c8d; }}
  .btn-plant {{ background:#27ae60; }}
  .btn-sky {{ background:#3498db; }}
  .btn-person {{ background:#f39c12; }}
  .btn-other {{ background:#555; }}
  .btn-skip {{ background:transparent; border:1px solid #555; }}
  .nav {{ display:flex; gap:15px; margin:20px 0; align-items:center; }}
  .nav-btn {{ padding:8px 16px; background:#333; border:none; border-radius:4px; color:#fff; cursor:pointer; }}
  .nav-btn:disabled {{ opacity:0.3; cursor:default; }}
  .count {{ font-size:16px; color:#e94560; }}
  .done {{ background:#27ae60; padding:2px 6px; border-radius:3px; font-size:11px; margin-left:5px; }}
  .keybinds {{ color:#666; font-size:12px; margin-top:10px; }}
</style>
</head>
<body>
<div class="top">
  <h1>Ground Truth 标注</h1>
  <span id="progressText" style="font-size:14px;">0/200</span>
  <span style="font-size:12px;color:#888;">快捷键: 1-8 选标签, ←→ 导航</span>
</div>
<div class="progress"><div class="progress-bar" id="progressBar" style="width:0%"></div></div>

<div class="main">
  <img class="patch" id="patch" src="" alt="Patch">
  <div class="info">
    图像: <span id="imgName">-</span> &nbsp;|&nbsp; 坐标: <span id="pxInfo">-</span> &nbsp;|&nbsp; SegFormer: <span id="segPred">-</span>
    <span id="labelDone" style="display:none;" class="done">已标注</span>
  </div>
  <div class="buttons">
    <button class="btn btn-wall" onclick="label('wall')">1 墙体 Wall</button>
    <button class="btn btn-building" onclick="label('building')">2 建筑 Building</button>
    <button class="btn btn-window" onclick="label('window')">3 窗户 Window</button>
    <button class="btn btn-door" onclick="label('door')">4 门 Door</button>
    <button class="btn btn-floor" onclick="label('floor')">5 地面 Floor</button>
    <button class="btn btn-plant" onclick="label('plant')">6 植物 Plant</button>
    <button class="btn btn-sky" onclick="label('sky')">7 天空 Sky</button>
    <button class="btn btn-person" onclick="label('person')">8 人 Person</button>
    <button class="btn btn-other" onclick="label('other')">0 其他 Other</button>
  </div>
  <div class="nav">
    <button class="nav-btn" onclick="prev()">← 上一个</button>
    <span class="count" id="counter">1 / 200</span>
    <button class="nav-btn" onclick="next()">下一个 →</button>
  </div>
  <div class="keybinds">键盘: 1=wall 2=building 3=window 4=door 5=floor 6=plant 7=sky 8=person 0=other ←→=导航</div>
  <button class="nav-btn" style="margin-top:15px;background:#27ae60;" onclick="exportLabels()">📥 导出标注结果</button>
</div>

<script>
const CLASSES = {classes_json};
const SAMPLES = {samples_json};
let currentIdx = 0;
let labels = {{}};

// 初始化: 从已有的 ground_truth 加载
SAMPLES.forEach((s, i) => {{
  if (s.ground_truth && s.ground_truth !== s.seg_prediction) {{
    labels[i] = s.ground_truth;
  }} else if (s.ground_truth) {{
    labels[i] = s.ground_truth;
  }}
}});

function show(idx) {{
  currentIdx = Math.max(0, Math.min(idx, SAMPLES.length - 1));
  const s = SAMPLES[currentIdx];
  document.getElementById('patch').src = 'patches/' + s.patch_file;
  document.getElementById('imgName').textContent = s.image_name;
  document.getElementById('pxInfo').textContent = '(' + s.px_u + ', ' + s.px_v + ')';
  document.getElementById('segPred').textContent = s.seg_prediction;
  document.getElementById('counter').textContent = (currentIdx + 1) + ' / ' + SAMPLES.length;

  const gt = labels[currentIdx];
  document.getElementById('labelDone').style.display = gt ? 'inline' : 'none';
  if (gt) document.getElementById('labelDone').textContent = '✓ ' + gt;

  const pct = Object.keys(labels).filter(k => labels[k] !== SAMPLES[parseInt(k)].seg_prediction).length;
  const total = SAMPLES.length;
  document.getElementById('progressText').textContent = 
    Object.keys(labels).length + '/' + total + ' 已标注 (' + pct + ' 已修改)';
  document.getElementById('progressBar').style.width = (Object.keys(labels).length / total * 100) + '%';
}}

function label(cls) {{
  labels[currentIdx] = cls;
  show(currentIdx);
  if (currentIdx < SAMPLES.length - 1) next();
}}

function next() {{ show(currentIdx + 1); }}
function prev() {{ show(currentIdx - 1); }}

function exportLabels() {{
  const out = JSON.parse(JSON.stringify(SAMPLES));
  out.forEach((s, i) => {{
    if (labels[i]) s.ground_truth = labels[i];
  }});
  const blob = new Blob([JSON.stringify({{samples: out, classes: CLASSES}}, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'labels_' + new Date().toISOString().slice(0,10) + '.json';
  a.click();
  alert('导出完成！将此文件覆盖 output/ground_truth/template.json 后运行 eval_ground_truth.py');
}}

// 键盘快捷键
document.addEventListener('keydown', e => {{
  const map = {{'1':'wall','2':'building','3':'window','4':'door','5':'floor','6':'plant','7':'sky','8':'person','0':'other'}};
  if (map[e.key]) {{ label(map[e.key]); e.preventDefault(); }}
  if (e.key === 'ArrowRight') {{ next(); e.preventDefault(); }}
  if (e.key === 'ArrowLeft') {{ prev(); e.preventDefault(); }}
}});

show(0);
</script>
</body>
</html>'''

out_path = PROJECT / 'output' / 'ground_truth' / 'labeling_tool.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"标注工具 → {out_path}")
print(f"用浏览器打开，1秒点1个标签，键盘1-8选类型，←→导航")
