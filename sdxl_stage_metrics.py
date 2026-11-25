import itertools,json,statistics
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize,InterpolationMode
from openai_clip_model import build_model
from transformers import CLIPTokenizer

@torch.inference_mode()
def main():
    root=Path('/data5/shenglin.zeng/sdxl_crossdistill');out=root/'outputs/hparam_pilot/teacher_stages'
    torch.set_num_threads(4)
    model=build_model(torch.jit.load(str(root/'models/metrics/ViT-B-32.pt'),map_location='cpu').state_dict()).float().eval()
    tokenizer=CLIPTokenizer.from_pretrained(root/'models/sdxl-base-1.0',subfolder='tokenizer',local_files_only=True)
    transform=Compose([Resize(224,interpolation=InterpolationMode.BICUBIC),CenterCrop(224),ToTensor(),Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    stages=[899,749,599,449];rows=[];features={};texts={}
    for rank in range(4):
        data=json.loads((out/f'rank{rank}.json').read_text());prompt=data['records'][0]['prompt']
        ids=tokenizer.encode(prompt,add_special_tokens=True,truncation=True,max_length=77)
        tokens=torch.zeros((1,77),dtype=torch.long);tokens[0,:len(ids)]=torch.tensor(ids)
        text=model.encode_text(tokens).float();texts[rank]=text/text.norm(dim=-1,keepdim=True)
        paths=[];keys=[]
        for rec in data['records']:
            for stage in rec['stages']:paths.append(stage['predicted_clean_image']);keys.append((rank,rec['seed'],stage['t']))
            paths.append(rec['final_image']);keys.append((rank,rec['seed'],-1))
        for a in range(0,len(paths),4):
            batch=torch.stack([transform(Image.open(p).convert('RGB')) for p in paths[a:a+4]])
            f=model.encode_image(batch).float();f=f/f.norm(dim=-1,keepdim=True)
            for key,value in zip(keys[a:a+4],f):features[key]=value
        print('STAGE_CLIP_DONE',rank,flush=True)
    for rank in range(4):
        for t in stages:
            alignment=[];similarity=[];diversity=[];final_div=[]
            for seed in [10,11,12,13]:
                f=features[(rank,seed,t)];ff=features[(rank,seed,-1)]
                alignment.append((f*texts[rank][0]).sum().item());similarity.append((f*ff).sum().item())
            for a,b in itertools.combinations([10,11,12,13],2):
                diversity.append(1-(features[(rank,a,t)]*features[(rank,b,t)]).sum().item())
                final_div.append(1-(features[(rank,a,-1)]*features[(rank,b,-1)]).sum().item())
            rows.append({'prompt_index':rank,'t':t,'clip_text_cosine':statistics.mean(alignment),'clip_similarity_to_final':statistics.mean(similarity),
                         'seed_pairwise_clip_distance':statistics.mean(diversity),'final_seed_pairwise_clip_distance':statistics.mean(final_div),
                         'diversity_ratio_to_final':statistics.mean(diversity)/statistics.mean(final_div)})
    summary={t:{k:statistics.mean(r[k] for r in rows if r['t']==t) for k in ['clip_text_cosine','clip_similarity_to_final','seed_pairwise_clip_distance','diversity_ratio_to_final']} for t in stages}
    (out/'stage_metrics.json').write_text(json.dumps({'model':'OpenAI CLIP ViT-B/32 FP32 CPU','prompts':4,'seeds':4,'rows':rows,'summary':summary,
        'interpretation':'Small teacher-only diagnostic. Intermediate x0 feature distance can reflect blur as well as semantics. No causal optimum or student quality claim.'},indent=2))
    print('STAGE_METRICS_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
