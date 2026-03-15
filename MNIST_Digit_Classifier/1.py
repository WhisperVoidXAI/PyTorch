import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. تحميل ومعالجة بيانات MNIST
transform = transforms.ToTensor()  # تحويل الصور إلى Tensor
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# 2. إنشاء DataLoader لتحميل البيانات على دفعات
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000)

# 3. تعريف شبكة عصبية بسيطة للتصنيف
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # طبقة مدخل → طبقة مخفية
        self.fc2 = nn.Linear(128, 10)     # طبقة مخفية → 10 فئات (الأرقام)

    def forward(self, x):
        x = x.view(-1, 28*28)   # تحويل الصورة إلى متجه (Flatten)
        x = F.relu(self.fc1(x)) # تفعيل ReLU
        return self.fc2(x)      # المخرجات النهائية (Logits)

# 4. إنشاء النموذج، دالة الخسارة، والمُحسّن
model = DigitClassifier()
criterion = nn.CrossEntropyLoss()           # لخسارة التصنيف متعدد الفئات
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. حلقة التدريب
for epoch in range(5):  # التدريب على 5 دورات (epochs)
    for images, labels in train_loader:
        outputs = model(images)               # تمرير البيانات للنموذج
        loss = criterion(outputs, labels)    # حساب الخسارة

        optimizer.zero_grad()  # إعادة تعيين التدرجات
        loss.backward()        # backpropagation
        optimizer.step()       # تحديث الأوزان

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 6. تقييم دقة النموذج على بيانات الاختبار
correct = 0
total = 0
with torch.no_grad():  # تعطيل حساب التدرجات أثناء الاختبار
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # اختيار الفئة الأعلى احتمالية
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')
"""""
ملاحظات:

Flattening: الصورة 28×28 تُحوّل لمتجه 784 بعد Flatten.

Activation Function: استخدمنا ReLU للطبقة المخفية.

CrossEntropyLoss: مناسبة لتصنيف متعدد الفئات.

DataLoader: يسمح بتحميل البيانات على دفعات لتسريع التدريب.

Evaluation: استخدام torch.no_grad() لتسريع الاختبار ومنع تحديث التدرجات.
"""