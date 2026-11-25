"""Build the isolated A×10 terminal PatchGAN arm from its actual trainer."""
from pathlib import Path
import json,hashlib
def prepare(parent, dest):
    s=parent.read_text()
    def replace(a,b):
        nonlocal s
        assert s.count(a)==1,a
        s=s.replace(a,b)
    replace("('sam','whole','none')","('hybrid','tile','d_only')")
    replace('Boundary joint4 expects SAM, whole-image decode, no augmentation','Boundary patch GAN expects hybrid, tile, D-only flips')
    replace("D_updates_per_G=1 if disc is not None else 0","D_updates_per_G=5 if disc is not None else 0")
    replace("spectral_norm_policy='not used by SAM heads'","spectral_norm_policy='refresh once per discriminator update'")
    replace("augmentation='whole-image SAM; no flips or latent crops'","augmentation='48 latent tile, center 256 RGB, D-only horizontal flips'")
    replace("    if rank == 0:\n        write_json(out / \"config.json\"",'''    config['student_gan_updates']='G3,G6,...,G159; terminal clean only'
    if rank == 0:
        write_json(out / "config.json"''')
    replace('    def pcm_loss(real, rnd, condition):','''    def discriminator_step(update):
        # Distinct full batch and rollout at every D update, independently of fake's index.
        positions = plan.rank_positions(update, rank)
        indices = [order[pos % len(order)] for pos in positions]
        cond = conditions(data, indices, device)
        rnd = random_inputs(positions, "fake", device, seed=args.seed, shape=latent_shape)
        with torch.no_grad():
            terminal = rollout_chunks(lambda c,n: rollout(c,n), cond, rnd["noise"], args.rollout_batch)
        real = data["latents"].index_select(0,torch.tensor(indices)).to(device,dtype=torch.float32)
        disc.set_heads(True)
        od.zero_grad(set_to_none=True)
        r1_on = (update+1)%16==0
        metric = torch.zeros(2,device=device)
        for start in range(0,len(indices),4):
            sl=slice(start,start+4)
            def view(z):
                return augment_view(decode_view(vae,z[sl],rnd["crops"][sl],kind='tile',chunk_size=1),rnd["flips"][sl],'D','d_only')
            with torch.no_grad():
                real_rgb=view(real)
                fake_rgb=view(terminal)
            real_rgb.requires_grad_(r1_on)
            txt=cond['pooled_prompt_embeds'][sl]
            pos=disc(real_rgb,txt)
            neg=disc(fake_rgb,txt)
            hinge=discriminator_loss(pos,neg,'hinge')
            r1=r1_penalty(pos,real_rgb) if r1_on else hinge.new_zeros(())
            scale=len(txt)/len(indices)
            ((hinge+.8*r1)*scale).backward()
            metric+=scale*torch.stack([hinge.detach(),r1.detach()])
        torch.nn.utils.clip_grad_norm_(disc.trainable(),10,error_if_nonfinite=True)
        od.step()
        od.zero_grad(set_to_none=True)
        disc.set_heads(False)
        record(dict(phase='D',D_update=update+1,D_hinge=metric[0].item(),R1_raw=metric[1].item(),R1_active=r1_on))

    def pcm_loss(real, rnd, condition):''')
    replace('        repeats = args.fake_updates','''        for _ in range(5):
            discriminator_step(dcount)
            dcount+=1
        repeats = args.fake_updates''')
    replace('train_d = disc is not None and (not boundary_mode or fake_repeat==repeats-1)','train_d = False  # D has its own five-update loop above')
    replace('use_gan = disc is not None if boundary_mode else stage in stage_plan.gan','use_gan = (gcount+1)%3==0')
    a=s.index('                            if disc is not None:',s.index('        use_gan ='))
    b=s.index('                        else:',s.index('                                gan_value = adversarial.detach()',a))
    s=s[:a]+'''                            if use_gan:
                                # Accumulate dGAN/dlatent in small image chunks, then backpropagate DM+GAN once.
                                gradient = torch.empty_like(generated)
                                for j in range(0,len(generated),2):
                                    sl=slice(j,j+2)
                                    leaf=generated[sl].detach().requires_grad_(True)
                                    rgb=decode_view(vae,leaf,rnd['crops'][sl],kind='tile',grad=True,chunk_size=1)
                                    value=generator_loss(disc(rgb,condition['pooled_prompt_embeds'][sl]),'hinge')*(len(leaf)/len(generated))
                                    gradient[sl]=torch.autograd.grad(value,leaf)[0]
                                    gan_value=gan_value+value.detach()
                                loss=loss+args.gan_weight*(generated.float()*gradient.detach()).sum()
                                if not torch.isfinite(gradient).all() or gradient.abs().max()==0:
                                    raise RuntimeError('Invalid terminal GAN gradient')
'''+s[b:]
    replace('                        if k==4 and disc is not None:\n                            del rgb, adversarial','                        if k==4 and use_gan:\n                            del rgb, gradient, leaf, value')
    compile(s,str(dest),'exec')
    dest.write_text(s)
if __name__=='__main__':
 import sys
 prepare(Path(sys.argv[1]),Path(sys.argv[2]))

