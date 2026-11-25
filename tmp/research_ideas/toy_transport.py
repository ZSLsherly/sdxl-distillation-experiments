"""Analytic CPU-only sanity checks. These are not diffusion/video results."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
a, b = 0.3, 0.7
deltas = np.array([0.2, 0.1, 0.05, 0.025])
rows = []
for delta in deltas:
    std = 1 + delta * a
    mean = delta * b
    stale_rmse = math.hypot(mean, 1 / std - std)
    pred_bias = 2 * a * b * delta**2
    pred_z_error = (-1 + 2 * a * delta) * std + 1 / std
    predicted_rmse = math.hypot(pred_bias, pred_z_error)
    rows.append(dict(delta=float(delta), new_mean=mean, new_std=std,
                     stale_rmse=stale_rmse, predicted_rmse=predicted_rmse,
                     rmse_ratio=stale_rmse / predicted_rmse))

stale = np.array([r['stale_rmse'] for r in rows])
predicted = np.array([r['predicted_rmse'] for r in rows])
stale_order = float(np.polyfit(np.log(deltas), np.log(stale), 1)[0])
predicted_order = float(np.polyfit(np.log(deltas), np.log(predicted), 1)[0])

def energy_distance(xs, px, ys, py):
    xs, px, ys, py = map(np.array, (xs, px, ys, py))
    cross = np.sum(np.abs(xs[:, None] - ys[None, :]) * px[:, None] * py[None, :])
    within_x = np.sum(np.abs(xs[:, None] - xs[None, :]) * px[:, None] * px[None, :])
    within_y = np.sum(np.abs(ys[:, None] - ys[None, :]) * py[:, None] * py[None, :])
    return float(2 * cross - within_x - within_y)

marginal_ed = energy_distance([-1, 1], [.5, .5], [-1, 1], [.5, .5])
conditional_ed = .5 * sum(energy_distance([-1, 1], [.5, .5], [h], [1.]) for h in [-1, 1])
assert marginal_ed == 0.0 and conditional_ed == 1.0
assert np.all(predicted < stale)
results = {
    'status': 'Analytic CPU toy; no trained image/video model and no empirical speedup.',
    'transport': {
        'old_distribution': 'N(0,1)',
        'transport': "Y=(1+delta*a)*X+delta*b", 'a': a, 'b': b,
        'true_score': '-(y-delta*b)/(1+delta*a)^2',
        'stale_score': '-y', 'predicted_score': '-y+delta*(2*a*y+b)',
        'evaluation_measure': 'Exact RMSE under the NEW Gaussian distribution.',
        'rows': rows,
        'loglog_fitted_order': {'stale': stale_order, 'predicted': predicted_order},
        'interpretation': 'Validates local first-order score-transport formula only. Old score and transport velocity are exact; no learned critic, approximation drift, or video complexity.'
    },
    'conditional_branch': {
        'prefix': 'H is -1 or +1 with probability 1/2.',
        'teacher_future': 'Y_T=-1 or +1 independently of H, each with probability 1/2.',
        'student_future': 'Y_S=H deterministically.',
        'energy_distance_convention': 'ED(P,Q)=2E|X-Y|-E|X-Xprime|-E|Y-Yprime|; no square root.',
        'marginal_energy_distance': marginal_ed,
        'mean_conditional_energy_distance': conditional_ed,
        'interpretation': 'Exact counterexample: matching future marginals can hide total within-prefix branching collapse. It is not a claim that all existing video losses are marginal-only.'
    }
}
(ROOT / 'toy_transport_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

# Pillow is available in the bundled runtime; matplotlib is not. Draw the
# analytic plot directly, keeping all numeric coordinates in this script.
img = Image.new('RGB', (1800, 800), 'white')
draw = ImageDraw.Draw(img)
font_file = 'C:/Windows/Fonts/msyh.ttc'
font = lambda n: ImageFont.truetype(font_file, n)
ink, muted, grid, teal, gray, orange = '#13263a', '#526477', '#e2e7ec', '#087f8c', '#7c8998', '#d0783c'
def label(x, y, s, size=23, fill=ink, anchor=None):
    draw.text((x, y), s, font=font(size), fill=fill, anchor=anchor)
label(70, 40, '两个解析反例：验证机制，不代表视频实验结果', 36)
label(70, 100, 'CPU 精确计算 · 未训练图像或视频模型 · 不含速度提升声明', 23, muted)
label(85, 171, 'A  训练分布漂移后的 score 误差', 28)
label(1010, 171, 'C  边际匹配会漏掉条件分支坍缩', 28)
label(138, 221, '新分布下的精确 RMSE（双对数坐标）', 22, muted)
label(1040, 221, 'Energy distance（不取平方根）', 22, muted)
x0, x1, y0, y1 = 145, 865, 615, 275
def xpos(d):
    return x0 + (math.log(d) - math.log(.02)) / (math.log(.25) - math.log(.02)) * (x1-x0)
def ypos(v):
    return y0 - (math.log(v) - math.log(.0002)) / (math.log(.3) - math.log(.0002)) * (y0-y1)
for tick in [.001, .01, .1]:
    yy = ypos(tick)
    draw.line((x0, yy, x1, yy), fill=grid, width=2)
    label(x0-18, yy, f'{tick:g}', 20, muted, anchor='rm')
for delta in deltas:
    xx=xpos(delta)
    draw.line((xx,y0,xx,y1), fill=grid, width=1)
    label(xx,y0+18,f'{delta:g}',20,muted,anchor='mt')
draw.line((x0,y1,x0,y0,x1,y0),fill=muted,width=2)
for vals,color in [(stale,gray),(predicted,teal)]:
    pts=[(xpos(float(d)),ypos(float(v))) for d,v in zip(deltas,vals)]
    draw.line(pts,fill=color,width=5)
    for xx,yy in pts:
        draw.ellipse((xx-7,yy-7,xx+7,yy+7),fill=color)
for yy,color,s in [(279,gray,f'旧 score：拟合阶数 {stale_order:.2f}'),
                   (316,teal,f'输运预测：拟合阶数 {predicted_order:.2f}')]:
    draw.line((164,yy+15,204,yy+15),fill=color,width=5)
    label(217,yy,s,21,color)
label((x0+x1)/2,670,'参数漂移尺度 δ',22,muted,anchor='mt')
rx0, rx1, ry0, ry1 = 1065, 1715, 615, 275
def ry(v): return ry0-v/1.2*(ry0-ry1)
for tick in [0,.5,1.]:
    yy=ry(tick)
    draw.line((rx0,yy,rx1,yy),fill=grid,width=2)
    label(rx0-18,yy,f'{tick:g}',20,muted,anchor='rm')
draw.line((rx0,ry1,rx0,ry0,rx1,ry0),fill=muted,width=2)
centers=[1220,1550]
draw.ellipse((centers[0]-7,ry0-7,centers[0]+7,ry0+7),fill=gray)
draw.rectangle((centers[1]-90,ry(1),centers[1]+90,ry0),fill=orange)
label(centers[0],ry0-45,'0',28,gray,anchor='mt')
label(centers[1],ry(1)-45,'1',28,orange,anchor='mt')
label(centers[0],635,'边际 ED',24,muted,anchor='mt')
label(centers[1],635,'同 prefix 的条件 ED',24,muted,anchor='mt')
label(85,739,'误差范数：一阶 → 二阶；旧 score 与漂移场在此均精确。',21,muted)
label(1010,729,'教师：每个 prefix 都有 ±1 两种未来。',21,muted)
label(1010,760,'学生：未来完全由 prefix 决定。',21,muted)
img.save(ROOT / 'toy_mechanism_checks.png')
print(json.dumps({'transport_rows': rows, 'orders': results['transport']['loglog_fitted_order'],
                  'marginal_ed': marginal_ed, 'conditional_ed': conditional_ed,
                  'plot': str(ROOT / 'toy_mechanism_checks.png')}, ensure_ascii=False, indent=2))
