import torch

input=torch.randn([1,1,6,6])
conv=torch.nn.Conv2d(1,1,1,1,[1,2])
out= conv(input)
print(out.detach().numpy())
print(out.shape)
pass