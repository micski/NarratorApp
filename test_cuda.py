import torch

# Sprawdzenie, czy GPU jest dostępne
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA jest dostępne! Używamy GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA nie jest dostępne, używamy CPU.")

# Prosty test z tensorem
x = torch.tensor([1.0, 2.0, 3.0], device=device)
print("Tensor na:", device, x)
