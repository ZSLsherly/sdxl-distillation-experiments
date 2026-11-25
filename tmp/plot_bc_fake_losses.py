"""Plot logged fake and student training diagnostics, using committed student cycles."""
from pathlib import Path
import json
import statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1] / "output/sdxl-crossdistill/a800_BC_paired_quota_20261102"
out = root / "loss_diagnostics"
colors = {"B":"#0072b2", "C":"#d55e00", "无先验":"#00845e"}
plt.rcParams.update({"font.family":"Microsoft YaHei", "axes.unicode_minus":False, "font.size":10})

def read_run(name, path):
    lines = path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines if line.endswith(b"\n")]
    student = [r for r in rows if r["phase"] == "student_done"]
    n = student[-1]["g"]
    fake = [r for r in rows if r["phase"] == "fake" and r["fake_update"] <= 2*n]
    assert [r["g"] for r in student] == list(range(1, n+1))
    assert [r["fake_update"] for r in fake] == list(range(1, 2*n+1))
    for i, g in enumerate(student):
        assert [r["fake_stage"] for r in fake[2*i:2*i+2]] == [g["stage"], 4]
        assert np.isfinite([g[k] for k in ("pcm_loss","terminal_dm_loss","middle_dm_loss","grad_norm")]).all()
    for f in fake:
        f["g"] = (f["fake_update"]+1)//2
        assert np.isfinite([f["fake_loss"], f["fake_grad_norm"]]).all()
    return name, student, fake

runs = []
for arm,name in [("B","B"),("C","C"),("uniform_control","无先验")]:
    path=root/"completed"/arm/"rank0.jsonl"
    if not path.exists():
        path=out/"uniform_snapshot.jsonl"
    runs.append(read_run(name,path))

def series(ax, rows, key, name):
    x=np.array([r["g"] for r in rows])
    y=np.array([r[key] for r in rows])
    color=colors[name]
    ax.scatter(x,y,color=color,alpha=.20,s=9,linewidth=0)
    means=[y[(x>=g-15)&(x<=g)].mean() for g in x]
    ax.plot(x,means,label=f"{name} · G{runs_by_name[name]}",color=color,
            ls="--" if name=="B" else "-",lw=1.6 if name=="B" else 2.2,
            zorder=4 if name=="B" else 3)

def decorate(ax,title):
    ax.set_title(title,loc="left")
    ax.set_xlabel("学生更新 G")
    ax.set_xlim(0,160)
    ax.grid(axis="y",alpha=.22)
    ax.spines[["top","right"]].set_visible(False)

runs_by_name={name:len(gs) for name,gs,fs in runs}
fig, axes=plt.subplots(2,3,figsize=(13,6.6),layout="constrained")
stage_names={2:"第2步 · x499",3:"第3步 · x249",4:"第4步 · 干净终点"}
for col,k in enumerate([2,3,4]):
    for i,key in enumerate(["fake_loss","fake_grad_norm"]):
        ax=axes[i,col]
        for name,gs,fs in runs:
            series(ax,[f for f in fs if f["fake_stage"]==k],key,name)
        decorate(ax,stage_names[k]+" | "+("fake 回归损失" if i==0 else "fake 裁剪前梯度范数"))
        ax.set_ylim(bottom=0)
axes[0,0].legend(frameon=False,fontsize=8)
fig.suptitle("fake 训练曲线    /    蓝虚线 B · 橙实线 C · 绿实线无先验    /    最近16个G内同阶段均值",fontsize=14)
fig.savefig(out/"fake_by_stage.png",dpi=155)
plt.close(fig)
fig, axes=plt.subplots(3,3,figsize=(13,8),layout="constrained",sharex=True)
gap={}
for col,k in enumerate((2,3,4)):
    paired=[]
    for i,(name,gs,fs) in enumerate(runs[:2]):
        subset=[r for r in fs if r["fake_stage"]==k]
        paired.append(subset)
        series(axes[i,col],subset,"fake_loss",name)
        decorate(axes[i,col],f"{name}   {stage_names[k]}   n={len(subset)}")
    top=max(axes[0,col].get_ylim()[1],axes[1,col].get_ylim()[1])
    for i in (0,1):axes[i,col].set_ylim(0,top)
    b,c=paired
    assert [r["fake_update"] for r in b]==[r["fake_update"] for r in c]
    x=np.array([r["g"] for r in b])
    diff=np.array([s["fake_loss"]-r["fake_loss"] for r,s in zip(b,c)])
    gap[k]={"observations_per_arm":len(diff),"B_mean":st.mean(r["fake_loss"] for r in b),
            "C_mean":st.mean(r["fake_loss"] for r in c),"paired_mean_abs_difference":float(np.abs(diff).mean())}
    ax=axes[2,col]
    ax.axhline(0,color="#777",lw=.8)
    ax.scatter(x,diff,s=9,alpha=.4,color="#73518b")
    ax.plot(x,[diff[(x>=g-15)&(x<=g)].mean() for g in x],color="#73518b")
    decorate(ax,f"C 减 B   {stage_names[k]}")
fig.suptitle("B、C 分行绘制，同列共用损失量程。底行为逐次配对差值，自用纵轴。",fontsize=14)
fig.savefig(out/"fake_bc_separate.png",dpi=155)
plt.close(fig)
(out/"paired_gap.json").write_text(json.dumps(gap,indent=2),encoding="utf-8")
fig,axes=plt.subplots(2,3,figsize=(13,6.6),layout="constrained")
for ax,(title,key,k) in zip(axes.flat,[
    ("首步 PCM","pcm_loss",None),("第2步 DM","middle_dm_loss",2),("第3步 DM","middle_dm_loss",3),
    ("末步 DM","terminal_dm_loss",None),("学生裁剪前梯度范数","grad_norm",None)]):
    for name,gs,fs in runs:series(ax,[g for g in gs if k is None or g["stage"]==k],key,name)
    decorate(ax,title)
axes[0,0].legend(frameon=False,fontsize=8)
ax=axes[1,2]
for arm,name in [("B","B"),("C","C"),("uniform_control","无先验")]:
    ev=[json.loads(p.read_text()) for p in sorted((root/"completed"/arm/"eval").glob("*/rank0.json"))]
    if not ev: continue
    ax.plot([e["student_updates"] for e in ev],[st.mean(x["latent_std"] for x in e["samples"]) for e in ev],".-",label=name,color=colors[name])
decorate(ax,"固定8张图的 latent 标准差均值（已归档节点）")
fig.suptitle("学生的训练表现    /    损失和梯度按学生更新 G 对齐",fontsize=14)
fig.savefig(out/"student_by_stage.png",dpi=155)
plt.close(fig)

summary={}
for name,gs,fs in runs:
    windows={}
    for lo,hi in [(1,min(runs_by_name.values())),(1,40),(41,80),(81,120),(121,160)]:
        if hi>len(gs):continue
        windows[f"{lo}-{hi}"]={}
        for k in [2,3,4]:
            part=[f for f in fs if lo<=f["g"]<=hi and f["fake_stage"]==k]
            windows[f"{lo}-{hi}"][str(k)]={"n":len(part),"loss":st.mean(f["fake_loss"] for f in part),"grad_norm":st.mean(f["fake_grad_norm"] for f in part)}
    summary[name]={"G":len(gs),"fake_updates":len(fs),"fake_clipped":sum(f["fake_grad_norm"]>1 for f in fs),
                   "max_fake_grad":max(f["fake_grad_norm"] for f in fs),"windows":windows}
(out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
common=min(runs_by_name.values())
table=""
for name,record in summary.items():
    for win,stages in record["windows"].items():
        table+="<tr>"+"".join(f"<td>{v}</td>" for v in [name,win,*[f'{stages[str(k)]["loss"]:.4f} ({stages[str(k)]["n"]})' for k in (2,3,4)]])+"</tr>"
report=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>fake 跟踪诊断</title>
<style>body{{max-width:1150px;margin:32px auto;padding:0 20px;font:16px/1.75 system-ui;color:#243749}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:5px 13px;border-bottom:1px solid #ddd}}</style>
<h1>fake 与学生的训练曲线</h1><p>B、C各到G{runs_by_name["B"]}，无先验组本次日志截至G{runs_by_name["无先验"]}。只纳入已完成学生更新对应的两次fake更新。每个点是64个样本的均值。三个组前G{common}可按同一训练进度比较。</p>
<img src="fake_by_stage.png"><p>每栏独立纵轴。第2、3步从已有噪声的边界继续加噪，末步从干净终点加噪，回归目标和难度不同。不能用跨栏数值大小判断谁学得好。</p>
<img src="fake_bc_separate.png"><p>B与C每组在第2、3、4步分别记录{gap[2]["observations_per_arm"]}、{gap[3]["observations_per_arm"]}、{gap[4]["observations_per_arm"]}次fake更新。原叠图先画B再画C，相近的中段曲线被覆盖。<a href="paired_gap.json">逐阶段配对统计</a></p>
<img src="student_by_stage.png"><p>DM曲线是经归一化、停止梯度的替代损失。固定样图latent标准差只描述输出尺度。淡点是原值，平滑窗口只含当前及之前的G。</p>
<h2>阶段均值</h2><p>括号内为该窗口对应阶段的fake更新数，原始数据见 <a href="summary.json">summary.json</a>。</p>
<table><tr><th>组</th><th>学生G区间</th><th>第2步</th><th>第3步</th><th>末步</th></tr>{table}</table>
<p>fake的loss是在本次fake参数更新之前，用本次训练批计算的加权噪声回归误差。它包含条件噪声的不可约误差，且受采样时间、损失权重和学生输出分布变化影响。中段整体平稳，B/C末步均值连续上升。当前日志足以定位末步，但不能据此判定fake滞后，也不能用两组末步loss的绝对值排名。</p>
<p>直接检验跟踪能力时，建议在固定学生快照上生成一批独立留出latent。为每个阶段固定提示词、初始噪声、合法t及加噪epsilon，在相同样本上比较当前fake和仅多做若干更新的诊断副本。额外更新用另一批样本。记录各时间段留出回归误差，以及由两份fake计算的学生DM方向夹角和范数比。报告多个快照的一致性。现有最终权重仅包含学生，这项结果不能从现有训练日志恢复。</p></html>"""
(out/"report.html").write_text(report,encoding="utf-8")
print(runs_by_name)
