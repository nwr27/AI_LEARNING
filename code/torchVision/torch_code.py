import torch
import torchvision
from torchvision import transforms

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

dataset = torchvision.datasets.ImageFolder("img", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

for batch in dataloader:
    images, labels = batch
    print(images.shape)
    print(labels)
    break
