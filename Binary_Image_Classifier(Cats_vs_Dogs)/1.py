import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# تعريف التحويلات للصور لمعالجة ما قبل التدريب (إعادة تغيير الحجم، التطبيع)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # إعادة تغيير حجم كل الصور إلى 128x128
    transforms.ToTensor(),          # تحويل الصور إلى تنسورات
    transforms.Normalize([0.5], [0.5])  # تطبيع قيم البكسل
])

# افتراض هيكل المجلدات: ./data/train/cat & ./data/train/dog
# كل مجلد يحتوي على الصور المقابلة
train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# تعريف شبكة CNN بسيطة للتصنيف الثنائي
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # الطبقة الالتفافية الأولى
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)               # تصغير الأبعاد
        self.fc1 = nn.Linear(32 * 32 * 32, 1)        # الطبقة المتصلة بالكامل للإخراج الثنائي

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 32 * 32 * 32)          # تسطيح قبل الطبقة المتصلة بالكامل
        return torch.sigmoid(self.fc1(x))     # دالة Sigmoid للإخراج الثنائي

model = CatDogCNN()

# دالة الخسارة للتصنيف الثنائي: Binary Cross-Entropy
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# حلقة التدريب
for epoch in range(5):
    for images, labels in train_loader:
        labels = labels.float().unsqueeze(1)  # تحويل إلى float وتغيير الشكل ليصبح [B, 1]
        outputs = model(images)               # التمرير الأمامي
        loss = criterion(outputs, labels)     # حساب الخسارة

        optimizer.zero_grad()                 # مسح التدرجات السابقة
        loss.backward()                       # تمرير خلفي
        optimizer.step()                      # تحديث الأوزان

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')