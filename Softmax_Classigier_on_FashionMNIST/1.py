import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# تحميل مجموعة بيانات FashionMNIST وتطبيق التحويلات
transform = transforms.ToTensor()  # تحويل الصور إلى تنسورات
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# إنشاء محملات البيانات
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# تعريف مصنف Softmax (بدون طبقات مخفية، من الإدخال إلى الإخراج مباشرة)
class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(28*28, 10)  # إدخال 28x28 -> 10 فئات

    def forward(self, x):
        x = x.view(-1, 28*28)              # تسطيح الصورة
        return self.linear(x)              # إخراج اللوغاريتمات الخام (Softmax يتم تطبيقه ضمنيًا في الخسارة)

model = SoftmaxClassifier()

# خسارة Cross-Entropy تقوم بتطبيق Softmax داخليًا
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# حلقة التدريب
for epoch in range(5):
    for images, labels in train_loader:
        outputs = model(images)              # التمرير الأمامي
        loss = criterion(outputs, labels)    # حساب الخسارة

        optimizer.zero_grad()                # مسح التدرجات السابقة
        loss.backward()                      # تمرير خلفي
        optimizer.step()                     # تحديث الأوزان

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# تقييم الدقة على مجموعة الاختبار
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # اختيار الفئة ذات اللوغاريتم الأقصى
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')