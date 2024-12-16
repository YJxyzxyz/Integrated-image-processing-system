import torch
data = torch.load('./models/AOD_net_epoch_relu_best.pth', map_location='cpu')
print(type(data))
