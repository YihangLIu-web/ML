import torch
from CNN import SimpleCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
x = torch.randn(64, 3, 224, 224)
model = SimpleCNN(num_classes=4)
model.to(device)

output = model(x)
print(output.shape)
