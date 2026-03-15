import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# خط المعالجة المسبقة لـ MobileNetV2
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # حجم الإدخال المطلوب لـ MobileNetV2
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # إحصاءات ImageNet
                         std=[0.229, 0.224, 0.225])
])

# تحميل مجموعة بيانات CIFAR-10
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# تحميل نموذج MobileNetV2 المدرَّب مسبقًا
model = models.mobilenet_v2(pretrained=True)

# استبدال طبقة المصنف للحصول على 10 فئات
model.classifier[1] = nn.Linear(model.last_channel, 10)

# تجميد طبقات استخراج الخصائص
for param in model.features.parameters():
    param.requires_grad = False

# دالة الخسارة والمُحسّن
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

# حلقة التدريب
for epoch in range(3):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# التقييم
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")