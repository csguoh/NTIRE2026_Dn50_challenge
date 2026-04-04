import sys
import math
import importlib.util
import os
import torch
import torch.nn as nn

sys.path.insert(0,"Restormer")
sys.path.insert(0,"NAFNet")

def _load_module_from_file(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_restormer(ckpt_path):

    arch_file="Restormer/basicsr/models/archs/restormer_arch.py"

    mod=_load_module_from_file("restormer_arch",arch_file)

    net=mod.Restormer(
        inp_channels=3,out_channels=3,dim=48,
        num_blocks=[4,6,6,8],
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="BiasFree",
        dual_pixel_task=False
    )

    ckpt=torch.load(ckpt_path,map_location="cpu")

    weights=ckpt.get("model",ckpt.get("params",ckpt.get("state_dict",ckpt)))
    weights={k.replace("module.",""):v for k,v in weights.items()}

    net.load_state_dict(weights,strict=True)

    return net


def load_nafnet(ckpt_path):

    arch_dir="NAFNet/basicsr/models/archs"

    for fname in ["arch_util.py","local_arch.py"]:
        fpath=os.path.join(arch_dir,fname)
        modname="basicsr.models.archs."+fname[:-3]
        if os.path.exists(fpath):
            spec=importlib.util.spec_from_file_location(modname,fpath)
            mod=importlib.util.module_from_spec(spec)
            sys.modules[modname]=mod
            spec.loader.exec_module(mod)

    arch_file="NAFNet/basicsr/models/archs/NAFNet_arch.py"

    mod=_load_module_from_file("nafnet_w64_arch",arch_file)

    net=mod.NAFNet(
        img_channel=3,
        width=64,
        middle_blk_num=12,
        enc_blk_nums=[2,2,4,8],
        dec_blk_nums=[2,2,2,2]
    )

    ckpt=torch.load(ckpt_path,map_location="cpu")

    weights=ckpt.get("model",ckpt.get("params",ckpt.get("params_ema",ckpt.get("state_dict",ckpt))))
    weights={k.replace("module.",""):v for k,v in weights.items()}

    net.load_state_dict(weights,strict=True)

    return net


def load_model_c(ckpt_path):

    arch_file="NAFNet/basicsr/models/archs/NAFNet_arch.py"

    mod=_load_module_from_file("nafnet_w32_arch",arch_file)

    net=mod.NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2,2,4,8],
        dec_blk_nums=[2,2,2,2]
    )

    ckpt=torch.load(ckpt_path,map_location="cpu")

    weights=ckpt.get("model",ckpt.get("params",ckpt.get("params_ema",ckpt.get("state_dict",ckpt))))
    weights={k.replace("module.",""):v for k,v in weights.items()}

    net.load_state_dict(weights,strict=True)

    return net


class TripleEnsemble(nn.Module):

    def __init__(self,restormer,nafnet,model_c,weights_init=(0.34,0.33,0.33)):
        super().__init__()

        self.restormer=restormer
        self.nafnet=nafnet
        self.model_c=model_c

        logits=torch.tensor([math.log(w+1e-8) for w in weights_init])
        self.logits=nn.Parameter(logits)

    def forward(self,x):

        w=torch.softmax(self.logits,dim=0)

        out_r=self.restormer(x)
        out_n=self.nafnet(x)
        out_c=self.model_c(x)

        return w[0]*out_r + w[1]*out_n + w[2]*out_c

    @property
    def weights(self):
        return torch.softmax(self.logits,dim=0).tolist()
