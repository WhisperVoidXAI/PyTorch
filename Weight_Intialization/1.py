import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. تحميل بيانات FashionMNIST وتحويلها لتنسورات
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 2. تعريف شبكة عصبية متعددة الطبقات مع تهيئة أوزان مخصصة
class InitNet(nn.Module):
    def __init__(self):
        super(InitNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)  # الطبقة الأولى
        self.fc2 = nn.Linear(256, 128)    # الطبقة الثانية
        self.out = nn.Linear(128, 10)     # طبقة المخرجات
        self._init_weights()              # استدعاء دالة تهيئة الأوزان

    # 3. دالة تهيئة الأوزان
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)   # Xavier initialization للطبقة الأولى
        nn.init.kaiming_normal_(self.fc2.weight)  # He initialization للطبقة الثانية
        nn.init.constant_(self.out.weight, 0.01)  # أوزان صغيرة ثابتة للمخرجات
        nn.init.zeros_(self.fc1.bias)             # تعيين الانحيازات للصفر
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.out.bias)

    # 4. دالة التمرير الأمامي
    def forward(self, x):
        x = x.view(-1, 28*28)           # تحويل الصورة لمتجه
        x = F.relu(self.fc1(x))         # تطبيق ReLU
        x = F.relu(self.fc2(x))         # تطبيق ReLU
        return self.out(x)              # مخرجات logits

# 5. إنشاء النموذج
model = InitNet()

# 6. تعريف الخسارة والمُحسّن
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7. حلقة التدريب
for epoch in range(5):  # تدريب لخمسة عصور
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()  # مسح التدرجات
        loss.backward()        # Backpropagation
        optimizer.step()       # تحديث الأوزان

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")