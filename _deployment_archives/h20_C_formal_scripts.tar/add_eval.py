from pathlib import Path
p=Path('train.py')
s=p.read_text()
a='                Image.fromarray(image).save(dest / filename)'
assert s.count(a)==1
s=s.replace(a,a+"""
                if gcount in (40,80,120,160):
                    diag=dest/'refinement'
                    diag.mkdir(exist_ok=True)
                    oid=data['original_indices'][idx]
                    one=conditions(data,[idx],device)
                    after=refine_for_gan(x,one,[oid],9100000)
                    after_rgb=vae.decode(after.float()/vae.config.scaling_factor,return_dict=False)[0]
                    if not torch.isfinite(after_rgb).all():
                        raise FloatingPointError('Nonfinite refinement evaluation')
                    after_img=Image.fromarray(((after_rgb[0]/2+.5).clamp(0,1).permute(1,2,0)*255).round().to(torch.uint8).cpu().numpy())
                    after_img.save(diag/('refined_'+filename))
                    full=Image.fromarray(image)
                    left=(full.width-299)//2
                    top=(full.height-299)//2
                    box=(left,top,left+299,top+299)
                    full.crop(box).save(diag/('four_crop_'+filename))
                    after_img.crop(box).save(diag/('refined_crop_'+filename))
""")
p.write_text(s)
