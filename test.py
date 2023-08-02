import torch
model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of parameters: {n_parameters}')
param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 0.00001}
]
optimizer = torch.optim.AdamW(param_dicts, 0.0001, weight_decay=0.0001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20)
epoch = 0
torch.save({'model':model.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': 0}, 'artifacts/train/Model.pth')
