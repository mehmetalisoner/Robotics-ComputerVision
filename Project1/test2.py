
# Example from pytorch https://pytorch.org/vision/stable/models.html

# WORKS but LOW CONFIDENCE
from torchvision.io import read_image
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights
import os
import matplotlib.pyplot as plt
from torchvision import transforms

# preprocess = transforms.Compose([            
#  transforms.Resize(256),                    
#  transforms.CenterCrop(224),                
#  transforms.ToTensor(),                     
#  transforms.Normalize(                      
#  mean=[0.485, 0.456, 0.406],                
#  std=[0.229, 0.224, 0.225]                  
# )])


folder = "test1"
filename = "balloon1.jpg"

img = read_image(os.path.join(folder,filename))
plt.imshow(img)
# Step 1: Initialize model with the best available weights
weights = ResNet50_QuantizedWeights.DEFAULT
model = resnet50(weights=weights, quantize=True)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score}%")