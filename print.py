import torch
N = torch.cuda.device_count()
for d in range(N):
    device = torch.device(f'cuda:{d}')
    free, total = torch.cuda.mem_get_info(device)
    mem_used_mb = (total - free) / 1024 ** 2
    print("device:",d,"mem:",mem_used_mb, "MB")