"""Summarize fresh teacher profiles; bootstrap prompts, not individual seeds."""
import argparse,json,html,csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def records(folder):
    result=[]
    for p in sorted(folder.glob('*.json')):
        d=json.loads(p.read_text(encoding='utf-8'))
        if 'rows' in d and 'mode' in d:result.append(d)
    return result


def array(data,grid,metric,field='rf_full'):
    curves=[]
    for r in data:
        points=[(v['u_mid'],v['fields'][field][metric]) for v in r['rows']
                if v['fields'][field].get('valid',True) and v['fields'][field].get(metric) is not None]
        x,y=zip(*sorted(points));curves.append(np.interp(grid,x,y))
    return np.array(curves)


def region(r,low,high,field='rf_full'):
    rows=[x for x in r['rows'] if low<=x['t_mid']<high and x['fields'][field].get('valid',True)]
    return sum(x['fields'][field]['angle_rad'] for x in rows)/sum(x['delta_u'] for x in rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    all_data=records(args.input);groups={k:[r for r in all_data if r['mode']==k] for k in
        ('base_dt10_bf16','fine_dt5_bf16','precision_dt10_fp32')}
    expected={'base_dt10_bf16':64,'fine_dt5_bf16':8,'precision_dt10_fp32':4}
    assert {k:len(v) for k,v in groups.items()}==expected,'Incomplete sampling; not publishing as final'
    base=groups['base_dt10_bf16']
    assert len({(r['prompt_index'],r['seed']) for r in base})==64
    assert len({r['prompt'] for r in base})==32,'Duplicate prompt strings need explicit accounting'
    umin=max(min(x['u_mid'] for x in r['rows']) for r in all_data)
    umax=min(max(x['u_mid'] for x in r['rows']) for r in all_data)
    grid=np.linspace(umin,umax,250)
    ref=sorted((x['u_mid'],x['t_mid']) for x in base[0]['rows'])
    native=np.interp(grid,*zip(*ref))
    rate=array(base,grid,'rate_rad_per_u')
    by_prompt=np.array([rate[[i for i,r in enumerate(base) if r['prompt_index']==p]].mean(0) for p in range(32)])
    rng=np.random.default_rng(20261026);indices=rng.integers(0,32,(1000,32))
    boot=by_prompt[indices].mean(1);ci=np.quantile(boot,[.025,.975],axis=0)
    rho=[float(spearmanr(grid,row).statistic) for row in rate]
    peaks=[max(r['rows'],key=lambda x:x['fields']['rf_full']['rate_rad_per_u'])['t_mid'] for r in base]
    areas={name:[region(r,lo,hi) for r in base] for name,lo,hi in
           [('early_800_980',800,980),('middle_400_600',400,600),('late_20_200',20,200)]}
    controls={}
    for name in ('fine_dt5_bf16','precision_dt10_fp32'):
        group=groups[name];keys={(r['prompt_index'],r['seed']) for r in group}
        paired=[r for r in base if (r['prompt_index'],r['seed']) in keys]
        original=array(paired,grid,'rate_rad_per_u').mean(0)
        controlled=array(group,grid,'rate_rad_per_u').mean(0)
        controls[name]=dict(trajectories=len(group),mean_curve_correlation=float(np.corrcoef(original,controlled)[0,1]),
            integrated_rate_ratio=float(np.trapz(controlled,grid)/np.trapz(original,grid)),
            peak_native_t=float(native[np.argmax(controlled)]))
    summary=dict(completed_trajectories=76,main_prompts=32,seeds=[201,202],cfg=8,resolution=1024,
        pointwise_95CI='bootstrap 32 prompts with their paired seeds, 1000 draws',
        median_trajectory_spearman_noise_vs_turning=float(np.median(rho)),
        fraction_trajectory_spearman_above_08=float(np.mean(np.array(rho)>.8)),
        fraction_peak_t_above_950=float(np.mean(np.array(peaks)>=950)),
        fraction_peak_t_below_50=float(np.mean(np.array(peaks)<=50)),
        region_mean_rates={k:float(np.mean(v)) for k,v in areas.items()},controls=controls,
        main_gpu_seconds=sum(r['seconds'] for r in base),
        peak_allocated_GiB_by_mode={k:max(r['peak_allocated_GiB'] for r in v) for k,v in groups.items()},
        SD3='deferred by user until model access is authorized')
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.out/'curves.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['u','native_t','mean_rate','prompt_bootstrap_low','prompt_bootstrap_high'])
        writer.writerows(zip(grid,native,rate.mean(0),ci[0],ci[1]))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    for ax,metric,title in zip(axes,['angle_rad','rate_rad_per_u','rate_rad_per_path_rms'],
        ['Angle per solver interval (rad)','Turning per physical noise u','Turning per RMS path distance']):
        arr=array(base,grid,metric)
        for row in arr[::6]:ax.plot(native,row,color='#aab8c3',alpha=.3,lw=.7)
        ax.plot(native,arr.mean(0),color='#176c9a',lw=2,label='New 64 trajectories')
        if metric=='rate_rad_per_u':ax.fill_between(native,ci[0],ci[1],color='#176c9a',alpha=.18,label='95% prompt bootstrap CI')
        ax.set(title=title,xlabel='SDXL timestep (early -> late)');ax.invert_xaxis();ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('SDXL teacher: 32 new prompts x 2 seeds | CFG 8 | 1024px',y=1.02)
    fig.tight_layout();fig.savefig(args.out/'turning_metrics.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for ax,(name,control) in zip(axes,controls.items()):
        subset=groups[name];keys={(r['prompt_index'],r['seed']) for r in subset}
        paired=[r for r in base if (r['prompt_index'],r['seed']) in keys]
        ax.plot(native,array(paired,grid,'rate_rad_per_u').mean(0),label='Matched dt10 BF16',lw=2)
        ax.plot(native,array(subset,grid,'rate_rad_per_u').mean(0),label=name,lw=2)
        ax.set(title=f"Matched control: r={control['mean_curve_correlation']:.3f}",xlabel='SDXL timestep',ylabel='rad / physical noise u')
        ax.invert_xaxis();ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.out/'controls.png',dpi=180);plt.close(fig)
    old_path=Path('output/sdxl-crossdistill/hparam_pilot/teacher_turning/result.json')
    if old_path.exists():
        old=json.loads(old_path.read_text(encoding='utf-8'))
        old_curve=np.interp(grid,old['u_grid'],old['modes']['base_dt10_bf16']['curves']['rf_full']['mean'])
        summary['old_new_mean_curve_correlation']=float(np.corrcoef(old_curve,rate.mean(0))[0,1])
        fig,ax=plt.subplots(figsize=(8,4.5))
        ax.plot(native,old_curve,label='Previous 16 prompts x 2 seeds');ax.plot(native,rate.mean(0),label='New 32 prompts x 2 seeds')
        ax.fill_between(native,ci[0],ci[1],alpha=.15);ax.invert_xaxis();ax.legend();ax.grid(alpha=.2)
        ax.set(xlabel='SDXL timestep (early -> late)',ylabel='rad / physical noise u',title='Independent prompt/seed extension')
        fig.tight_layout();fig.savefig(args.out/'old_vs_new.png',dpi=180);plt.close(fig)
        (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    imgs=''.join(f'<figure><img src="raw/example_prompt{i:03d}_seed{seed}.png"><figcaption>Prompt {i}, seed {seed}: {html.escape(base[2*i]["prompt"])}</figcaption></figure>' for i in range(4) for seed in (201,202))
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>SDXL教师转向扩样</title>
<style>body{{font:16px/1.8 "Microsoft YaHei",sans-serif;background:#f3f6f8;color:#223546}}main{{max-width:1150px;margin:30px auto;background:white;padding:30px}}img{{max-width:100%}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}figure{{margin:0}}figcaption{{font-size:12px}}pre{{white-space:pre-wrap;background:#eef4f7;padding:16px}}a{{color:#176c9a}}</style><main>
<h1>SDXL教师转向：新增64条轨迹与12条对照</h1><p>2026-10-26 · 真实冻结SDXL采样 · 32条新增提示词，seed201/202，CFG8，1024分辨率。SD3按用户决定待授权后补充。</p>
<p>主统计为物理噪声坐标u中的转向率；同时提供原始转角和RF等价状态y中的路径距离归一化。三个量不能混称。路径距离使用latent元素RMS，绝对值取决于状态空间尺度。</p>
<img src="turning_metrics.png"><p>浅色线为部分单轨迹，阴影为按提示词成组bootstrap的逐点95%区间。横轴从高噪声到低噪声；计算未平滑曲线。</p>
<img src="old_vs_new.png"><img src="controls.png">
<p>步长对照8条，FP32运算对照4条，均匹配主样本。FP32和BF16加载同一FP16权重；TF32关闭。低噪声末端未把“scheduler索引0”当成零噪声。</p>
<pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre>
<p>这次测量不直接证明任何训练采样策略有效。高噪声总体偏高与存在可泛化的中间转折点是两种结论；边界处的最大值不能定位独特阶段分界。</p>
<p><a href="summary.json">统计JSON</a> · <a href="curves.csv">曲线CSV</a> · <a href="metric_cpu_check.json">解析圆轨迹指标验证</a></p>
<h2>部分实际采样图像</h2><div class="grid">{imgs}</div></main></html>'''
    (args.out/'report.html').write_text(body,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
