import torch
import os
from collections import OrderedDict
from unet3d import UNet3D
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#if isinstance(model,torch.nn.DataParallel):
#torch.save(model.module.state_dict(),config.save_path)
def to_pt(model_path="./output/v_norm17/v_Norm17.pth"):
    model = UNet3D()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    model.load_state_dict(
        OrderedDict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()}))
    model.eval()
    example = torch.rand(1, 1, 32, 160, 160).to(device=device)
    with torch.no_grad():
        trace_script_module = torch.jit.trace(model, example)
        trace_script_module.save("./output/v_norm17/v_norm17.pt")

if __name__=="__main__":
    to_pt()