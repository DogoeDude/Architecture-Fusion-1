import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# 1. Dataset (CIFAR-10)
# ------------------------
transform_train = transforms.Compose([
    transforms.Resize((64,64)),  # upscale for transformer
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# ------------------------
# 2. CNN + Transformer Fusion Model
# ------------------------
class CNNTransformerFusion(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Pretrained CNN backbone
        cnn = models.resnet18(weights="DEFAULT")
        self.cnn = nn.Sequential(*list(cnn.children())[:-2])  # remove FC + avgpool
        self.cnn_out_channels = 512

        # Project CNN output to transformer embedding
        self.conv1x1 = nn.Conv2d(self.cnn_out_channels, 256, kernel_size=1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classification head
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)                # [B, C, H, W]
        x = self.conv1x1(x)            # [B, 256, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        x = self.transformer(x)        # [B, H*W, C]
        x = x.mean(dim=1)              # global average pooling
        out = self.classifier(x)       # [B, num_classes]
        return out

# ------------------------
# 3. Model, Loss, Optimizer
# ------------------------
model = CNNTransformerFusion(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# ------------------------
# 4. Training Loop
# ------------------------
num_epochs = 10
train_acc_history = []
test_acc_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100*correct/total:.2f}%"})

    train_acc = 100 * correct / total
    train_acc_history.append(train_acc)

    # ------------------------
    # 5. Validation Accuracy
    # ------------------------
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)
    test_acc = 100 * correct_test / total_test
    test_acc_history.append(test_acc)
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

# ------------------------
# 6. Plot Accuracy Curve
# ------------------------
plt.figure(figsize=(8,5))
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(test_acc_history, label='Test Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training & Test Accuracy")
plt.legend()
plt.show()

# ------------------------
# 7. Sample Predictions
# ------------------------
classes = train_dataset.classes
model.eval()
images, labels = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    outputs = model(images)
    _, preds = outputs.max(1)

plt.figure(figsize=(12,6))
for i in range(8):
    img = images[i].cpu().permute(1,2,0).numpy()
    img = (img * 0.5 + 0.5)  # denormalize
    plt.subplot(2,4,i+1)
    plt.imshow(img)
    plt.title(f"Pred: {classes[preds[i]]}\nTrue: {classes[labels[i]]}")
    plt.axis('off')
plt.show()
