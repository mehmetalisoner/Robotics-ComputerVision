# creating custom dataset, multiple directories solution
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

resnet50 = models.resnet50(pretrained=True)

class CustomDataset(Dataset):
	def __init__(self,imgs_path,class_map,img_dim):
		self.imgs_path = imgs_path
		file_list = glob.glob(self.imgs_path + "*")
		print(file_list)
		self.data = []
		for class_path in file_list:
			class_name = class_path.split("/")[-1]
			for img_path in glob.glob(class_path + "/*.jpg"):
				self.data.append([img_path, class_name])
		print(self.data)
		self.class_map = class_map
		self.img_dim = img_dim
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path, class_name = self.data[idx]
		img = cv2.imread(img_path)
		img = cv2.resize(img, self.img_dim)
		class_id = self.class_map[class_name]
		img_tensor = torch.from_numpy(img)
		img_tensor = img_tensor.permute(2, 0, 1)
		class_id = torch.tensor([class_id])
		return img_tensor, class_id


img_folder = "dataset/"
class_map = {"balloon":0, "sports car":1, "bee":2, "labrador retriever":3}
img_dim = (416,416)

dataset = CustomDataset(imgs_path=img_folder,class_map=class_map,img_dim=img_dim)		
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
for imgs, labels in data_loader:
    print("Batch of images has shape: ",imgs.shape)
    print("Batch of labels has shape: ", labels.shape)

with torch.no_grad():
    output = resnet50(data_loader)


probabilities = torch.nn.functional.softmax(output[0], dim=0)


    # Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
print("New item:")
y_pred.append(categories[top5_catid[0]]) # Save Prediction
#y_true.extend()

print(results)

