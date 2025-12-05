import torch

from PIL import Image
from network.VGG16 import VGG16
from torchvision.transforms import transforms
tra = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img = Image.open('c.jpg').convert('RGB')
img.show()
img = tra(img)
img = torch.reshape(img,(1,3,224,224))
vgg = VGG16()
vgg.load_state_dict(torch.load('vgg.pth',weights_only=False))
vgg.eval()

with torch.no_grad():
    output = vgg(img)
    print(output)
print(vgg)