import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. تعريف التحويلات (EMNIST الصور مائلة)
transform = transforms.Compose([
    transforms.RandomRotation((-90, -90)),  # تدوير لتصبح الصورة منتصبة
    transforms.ToTensor()
])

# 2. تحميل بيانات EMNIST Balanced
train_data = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
test_data = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# 3. تعريف نموذج CNN للتعرف على الحروف
class EMNISTClassifier(nn.Module):
    def __init__(self):
        super(EMNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 47)  # 47 فئة للحروف والأرقام

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # الطبقة الأولى + MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # الطبقة الثانية + MaxPool
        x = x.view(-1, 64 * 7 * 7)           # تحويل الخرائط إلى متجه
        x = F.relu(self.fc1(x))               # الطبقة المخفية
        return self.fc2(x)                    # مخرجات logits للفئات 47

# 4. إنشاء النموذج
model = EMNISTClassifier()

# 5. تعريف الخسارة والمُحسّن
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. حلقة التدريب
for epoch in range(5):  # تدريب 5 عصور
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()  # مسح التدرجات
        loss.backward()        # Backpropagation
        optimizer.step()       # تحديث الأوزان

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 7. التقييم على بيانات الاختبار
correct = 0
total = 0
with torch.no_grad():  # تعطيل الحسابات التدرجية أثناء الاختبار
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # اختيار الفئة ذات أعلى قيمة
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")