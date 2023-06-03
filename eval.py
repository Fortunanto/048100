import torch
import torch.nn as nn
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
from torchvision import models
from tqdm import tqdm

# Load your trained model
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes for gender prediction
model.load_state_dict(torch.load('final_model.pt'))
model.eval()

if torch.cuda.is_available():
    model = model.cuda()

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Read CSV file
df = pd.read_csv('fairface_label_val.csv')

# For each image in the CSV file
for index, row in tqdm(df.iterrows(),total=df.shape[0]):
    img_path = row["file"]
    img = Image.open(img_path).convert('RGB')  # Convert image to RGB
    img = transform(img).unsqueeze(0)  # Apply transformations and add batch dimension

    if torch.cuda.is_available():
        img = img.cuda()

    # Predict gender
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
        pred = 'Female' if pred.item() == 0 else 'Male'  # Adjust based on your class index mapping

    # Add to DataFrame
    df.loc[index, 'predicted_gender'] = pred

# Save DataFrame to CSV
df.to_csv('fairface_label_val_expanded.csv', index=False)
