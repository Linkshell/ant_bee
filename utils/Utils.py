from torchvision.transforms import transforms
from PIL import Image

def To_PIL_Image(img):
    pil_transform = transforms.ToPILImage()
    img = pil_transform(img)
    img.show()