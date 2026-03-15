import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 1. المعالجة المسبقة للصور لتناسب GoogLeNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # الحجم المطلوب للنموذج المدرب على ImageNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 2. تحميل بيانات CIFAR-10
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# 3. تحميل نموذج GoogLeNet المدرب مسبقًا
model = models.googlenet(pretrained=True)

# 4. استبدال طبقة التصنيف النهائية
model.fc = nn.Linear(model.fc.in_features, 10)

# 5. تجميد جميع الطبقات عدا الطبقة الأخيرة
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

# 6. تعريف دالة الخسارة والمُحسّن
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# 7. حلقة التدريب (تدريب الطبقة الأخيرة فقط)
for epoch in range(3):
    for images, labels in train_loader:

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 8. تقييم النموذج
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")