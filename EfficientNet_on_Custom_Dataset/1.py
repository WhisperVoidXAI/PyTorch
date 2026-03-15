import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 1. المعالجة المسبقة للصور
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # الحجم المطلوب لـ EfficientNet
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # تطبيع ImageNet
        std=[0.229, 0.224, 0.225]
    )
])

# 2. تحميل البيانات المخصصة
train_data = datasets.ImageFolder(
    root='./custom_data/train',
    transform=transform
)

test_data = datasets.ImageFolder(
    root='./custom_data/test',
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 3. تحميل نموذج EfficientNet-B0 المدرب مسبقًا
model = models.efficientnet_b0(pretrained=True)

# 4. استبدال طبقة التصنيف الأخيرة
num_classes = len(train_data.classes)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

# 5. تجميد الطبقات الأساسية
for param in model.features.parameters():
    param.requires_grad = False

# 6. تعريف دالة الخسارة والمُحسّن
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

# 7. حلقة التدريب
for epoch in range(5):

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