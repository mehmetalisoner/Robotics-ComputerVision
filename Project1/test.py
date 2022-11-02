# Works best, example from https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/



import torch
import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image
import os


resnet50 = models.resnet50(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([            
 transforms.Resize(256),                    
 transforms.CenterCrop(224),                
 transforms.ToTensor(),                     
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
)])


folder = "test1"
filename = "blibla.jpg"
img_name = os.path.join(folder,filename)
img = Image.open(img_name) 
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
resnet50.eval()
out = resnet50(batch_t)
with open('imagenet_classes.txt') as f:
  classes = [line.strip() for line in f.readlines()]


_, index = torch.max(out, 1)
 
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
 
print(classes[index[0]], percentage[index[0]].item())
_, indices = torch.sort(out, descending=True)
[(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]  