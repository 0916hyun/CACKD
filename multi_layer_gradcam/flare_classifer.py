import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd

class PairedDataset(Dataset):
    def __init__(self, flare_img_dir, clean_img_dir, transform=None):
        self.flare_img_dir = flare_img_dir
        self.clean_img_dir = clean_img_dir
        self.flare_img_files = sorted([f for f in os.listdir(flare_img_dir) if f.endswith('.png')])
        self.clean_img_files = sorted([f for f in os.listdir(clean_img_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.flare_img_files)

    def __getitem__(self, idx):
        flare_img_name = self.flare_img_files[idx]
        clean_img_name = self.clean_img_files[idx]
        flare_img_path = os.path.join(self.flare_img_dir, flare_img_name)
        clean_img_path = os.path.join(self.clean_img_dir, clean_img_name)

        flare_image = Image.open(flare_img_path).convert('RGB')
        clean_image = Image.open(clean_img_path).convert('RGB')

        if self.transform:
            flare_image = self.transform(flare_image)
            clean_image = self.transform(clean_image)

        return flare_image, clean_image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

flare_img_dir = ''
clean_img_dir = ''
valid_clean_img_dir = ''
valid_flare_img_dir = ''
save_dir = 'weight dir'
os.makedirs(save_dir, exist_ok=True)

train_dataset = PairedDataset(flare_img_dir, clean_img_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

if valid_flare_img_dir and valid_clean_img_dir:
    valid_dataset = PairedDataset(valid_flare_img_dir, valid_clean_img_dir, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
else:
    valid_loader = None

model = models.vgg16(pretrained=True)

model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 2)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def evaluate(loader):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    for flare_inputs, clean_inputs in loader:
        inputs = torch.cat((flare_inputs, clean_inputs), dim=0)
        labels = torch.cat((torch.ones(flare_inputs.size(0)), torch.zeros(clean_inputs.size(0))), dim=0).long()
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
    loss = running_loss / total_samples
    accuracy = running_corrects.double() / total_samples
    return loss, accuracy

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=30):
    best_acc = 0.0
    best_epoch = 0
    
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for flare_inputs, clean_inputs in tqdm(train_loader):
            inputs = torch.cat((flare_inputs, clean_inputs), dim=0)
            labels = torch.cat((torch.ones(flare_inputs.size(0)), torch.zeros(clean_inputs.size(0))), dim=0).long()
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
        
        train_loss = running_loss / total_samples
        train_acc = running_corrects.double() / total_samples
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')

        if valid_loader:
            val_loss, val_acc = evaluate(valid_loader)

            print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
            
            history["epoch"].append(epoch+1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc.item())
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc.item())

        torch.save(model.state_dict(), os.path.join(save_dir, f'densenet169_epoch_{epoch+1}.pth'))

    print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch}')
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)

    return model

model = train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=30)
