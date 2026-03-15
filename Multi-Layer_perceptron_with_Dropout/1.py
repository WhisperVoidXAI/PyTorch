import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. تحميل FashionMNIST وتطبيق التحويلات
transform = transforms.ToTensor()  # تحويل الصور إلى Tensor
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

# تحميل البيانات على دفعات
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000)

# 2. تعريف MLP مع Dropout
class MLPDropout(nn.Module):
    def __init__(self):
        super(MLPDropout, self).__init__()
        self.fc1   = nn.Linear(28*28, 256)  # الطبقة المخفية الأولى
        self.drop1 = nn.Dropout(0.5)        # Dropout بعد الطبقة الأولى
        self.fc2   = nn.Linear(256, 128)    # الطبقة المخفية الثانية
        self.drop2 = nn.Dropout(0.5)        # Dropout بعد الطبقة الثانية
        self.out   = nn.Linear(128, 10)     # طبقة المخرجات لعشر فئات

    def forward(self, x):
        x = x.view(-1, 28*28)       # تسطيح الصورة
        x = F.relu(self.fc1(x))     # ReLU
        x = self.drop1(x)           # تطبيق Dropout
        x = F.relu(self.fc2(x))     # ReLU
        x = self.drop2(x)           # تطبيق Dropout
        return self.out(x)          # مخرجات logits

model = MLPDropout()

# 3. تعريف الخسارة والمُحسّن
criterion = nn.CrossEntropyLoss()           # خسارة التصنيف متعدد الفئات
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. حلقة التدريب
for epoch in range(5):
    for images, labels in train_loader:
        outputs = model(images)             # تمرير البيانات
        loss = criterion(outputs, labels)  # حساب الخسارة

        optimizer.zero_grad()               # إعادة تعيين التدرجات
        loss.backward()                     # backpropagation
        optimizer.step()                    # تحديث الأوزان

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 5. التقييم على بيانات الاختبار
correct = 0
total = 0
with torch.no_grad():                       # تعطيل التدرجات أثناء الاختبار
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy with Dropout: {100 * correct / total:.2f}%')