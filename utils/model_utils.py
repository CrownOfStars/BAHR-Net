import torch.nn as nn
import torch
import torch.distributed as dist

def freeze_module(model,keys):
    for name, para in model.named_parameters():
        for key in keys:
            if key in name:
                para.requires_grad_(False)
                
def defreeze_all(model:nn.Module):
    for _, para in model.named_parameters():
        if not para.requires_grad:
            para.requires_grad_(True)

def clip_gradient(model, max_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

def get_allreduce_avg(count,data):
    result = {}
    count = torch.tensor(count, dtype=torch.float32).cuda()
    dist.all_reduce(count, op=dist.ReduceOp.SUM)

    for k,v in data.items():
        dist.all_reduce(v,op = dist.ReduceOp.SUM)
        result[k] = v.item()/count.item()
    return result

def safe_load_model(model,state_dict,ckpt_path):
    
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(ckpt_path, msg))
