from pathlib import Path
s=Path('sdxl_gan_evaluate.py').read_text(encoding='utf-8')
s=s.replace("OUT=ROOT/'outputs/gan_stage_pilot'","OUT=ROOT/'outputs/gan_stage_pilot_v2'").replace("'joint34'","'joint34_matched'")
s=s.replace("'joint34_matched':16","'joint34_matched':8")
s=s.replace("            assert any(r.get('r1',0)>0 for r in d)","            if trial!='dmd':assert any(r.get('r1',0)>0 for r in d)\n            else:assert all(r['d_fake_stage'] is None and r['d_decode_seconds']==r['d_update_seconds']==0 for r in d)\n            assert json.loads((OUT/trial/'execution_audit.json').read_text())['status']=='passed'")
s=s.replace("assert rank_summary['student_parameter_delta']>0","assert rank_summary['student_parameter_delta']>0 and rank_summary['status']=='training_complete'")
s=s.replace("statistics.median(r['seconds'] for r in student)","statistics.median(r['seconds'] for r in student if r['iteration']//6>=2)")
s=s.replace("statistics.median(r['seconds'] for r in critic)","statistics.median(r['seconds'] for r in critic if r['iteration']>=2)")
needle="    assert len({json.dumps(audit['trials'][t]['source_sha256'],sort_keys=True) for t in TRIALS})==1"
s=s.replace(needle,"""        audit['trials'][trial]['timing_breakdown_median_seconds']={k:statistics.median(r[k] for r in critic if r['iteration']>=2) for k in ['rollout_seconds','d_decode_seconds','d_update_seconds','fake_score_seconds']}
        audit['trials'][trial]['timing_warmup']='Exclude first two student and first two fake updates per rank; overall train_seconds includes all updates'
        audit['trials'][trial]['execution_audit']=json.loads((OUT/trial/'execution_audit.json').read_text())
"""+needle)
s=s.replace("    for trial in TRIALS:\n        disc=PixelPatchDiscriminator()","    for trial in TRIALS:\n        if trial=='dmd':continue\n        disc=PixelPatchDiscriminator()")
s=s.replace("scores={'train_real':[],'heldout_real':[],'heldout_fake':[]}","scores={'train_real':[],'heldout_real':[],'heldout_fake':[],'heldout_fake_stage3':[]}")
needle="                scores['heldout_fake'].append(disc(decode_tile(vae,latent,(40,40)),condition).item())"
s=s.replace(needle,needle+"\n                x3=mapping[trial][(idx,101)]['clean_estimates'][2].to('cuda',dtype=torch.float32)\n                scores['heldout_fake_stage3'].append(disc(decode_tile(vae,x3,(40,40)),condition).item())")
s=s.replace("'joint34 has twice the GAN exposure count at the same weight.'","'joint34_matched and joint4 have equal GAN update counts and coefficients; stage distributions and gradient magnitudes can still differ.'")
Path('sdxl_gan_evaluate_v2.py').write_text(s,encoding='utf-8')
s=Path('sdxl_gan_render_report.py').read_text(encoding='utf-8').replace("outputs/gan_stage_pilot'","outputs/gan_stage_pilot_v2'").replace("'joint34'","'joint34_matched'")
s=s.replace('D：末两步DMD+GAN','D：末两步GAN（次数匹配）')
s=s.replace("for trial in trials:\n    d=result['discriminator_train_holdout_diagnostics'][trial]","for trial in trials:\n    if trial=='dmd':continue\n    d=result['discriminator_train_holdout_diagnostics'][trial]")
s=s.replace("<td>{d[\"heldout_fake\"]:.4f}</td>","<td>{d[\"heldout_fake_stage3\"]:.4f}</td><td>{d[\"heldout_fake\"]:.4f}</td>")
s=s.replace('每组24次学生更新、120次fake score与判别器更新','每组24次学生更新、120次fake score更新；A组0次判别器更新，其余组各120次')
s=s.replace('D组GAN出现次数是C组两倍；','C、D组GAN学生更新均为8次，D在阶段3/4各4次；')
s=s.replace('A组也训练隔离的判别器作诊断。','A组完全跳过GAN解码和判别器，下面分项计时给出实际开销。计时中位数排除了每卡前两次学生及前两次fake更新。')
s=s.replace('<th>留出生成logit</th>','<th>留出阶段3 logit</th><th>留出阶段4 logit</th>')
s=s.replace('href="../gan_stage_protocol.html"','href="../reviews/gan_review_resolution_20261017.html"')
needle="(ROOT/'report.html').write_text(body,encoding='utf-8')"
extra="""timing_rows=[]
for t in trials:
    m=metrics[t]['training']['timing_breakdown_median_seconds']
    timing_rows.append('<tr><td>'+names[t]+'</td>'+''.join(f'<td>{m[k]:.3f}s</td>' for k in ['rollout_seconds','d_decode_seconds','d_update_seconds','fake_score_seconds'])+'</tr>')
body=body.replace('<h2>判别器过拟合诊断</h2>','<h2>辅助更新分项计时</h2><table><tr><th>组</th><th>学生rollout</th><th>D解码</th><th>D训练</th><th>fake score训练</th></tr>'+''.join(timing_rows)+'</table><p>各项分别取中位数，不能保证中位数之和等于整体中位数。GAN学生更新还含VAE输入梯度探针，当前吞吐包括诊断成本。</p><h2>判别器过拟合诊断</h2>')
"""
s=s.replace(needle,extra+needle)
Path('sdxl_gan_render_report_v2.py').write_text(s,encoding='utf-8')
print('WROTE v2 evaluation and report')
