"""Split SAM global attention by query row. Every query sees all keys and all 32 images stay in the batch."""
from types import MethodType
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def row_attention(self, hidden_states, output_attentions=False):
    x=hidden_states
    if output_attentions:raise ValueError('The discriminator uses features only')
    b,h,w,_=x.shape
    q,k,v=self.qkv(x).reshape(b,h*w,3,self.num_attention_heads,-1).permute(2,0,3,1,4).unbind(0)
    rh=self.get_rel_pos(h,h,self.rel_pos_h)
    rw=self.get_rel_pos(w,w,self.rel_pos_w)
    def stripe(qs,k,v,rh_slice,rw):
        qr=qs.reshape(b,self.num_attention_heads,-1,w,qs.size(-1))
        y=torch.einsum('bnijd,ikd->bnijk',qr,rh_slice)
        z=torch.einsum('bnijd,jkd->bnijk',qr,rw)
        bias=(y.unsqueeze(-1)+z.unsqueeze(-2)).contiguous().reshape(b,self.num_attention_heads,qs.size(2),h*w)
        return F.scaled_dot_product_attention(qs,k,v,attn_mask=bias)
    parts=[]
    for i in range(0,h,2):
        end=min(h,i+2);args=(q[:,:,i*w:end*w],k,v,rh[i:end],rw)
        parts.append(checkpoint(stripe,*args,use_reentrant=False) if x.requires_grad else stripe(*args))
    a=torch.cat(parts,2).transpose(1,2).reshape(b,h,w,-1)
    return self.proj(a),None

def install_rows(backbone):
    for i in backbone.config.global_attn_indexes:
        att=backbone.layers[i].attn
        assert att.use_rel_pos
        att.forward=MethodType(row_attention,att)

def backbone_features(backbone,x):
    x=backbone.patch_embed(x)
    if backbone.pos_embed is not None:x=x+backbone.pos_embed
    features=[]
    for i,layer in enumerate(backbone.layers,1):
        x=checkpoint(layer,x,use_reentrant=False) if x.requires_grad else layer(x)
        if i in (3,6,9,12):features.append(x.permute(0,3,1,2).contiguous())
    return tuple(features)
