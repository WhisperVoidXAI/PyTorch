import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. تفعيل CUDA إذا كان متاحًا
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. المعالجة المسبقة للبيانات
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 3. تحميل بيانات CIFAR-10
train_data = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

# 4. تعريف نموذج CNN بسيط
class SmallCNN(nn.Module):

    def __init__(self):
        super(SmallCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 16 * 16)

        x = F.relu(self.fc1(x))

        return self.fc2(x)

# 5. نقل النموذج إلى الجهاز
model = SmallCNN().to(device)

# 6. تعريف دالة الخسارة والمُحسّن
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7. إنشاء GradScaler الخاص بـ AMP
scaler = torch.cuda.amp.GradScaler()

# 8. حلقة التدريب باستخدام Mixed Precision
for epoch in range(3):

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # تشغيل الحساب بالدقة المختلطة
        with torch.cuda.amp.autocast():

            outputs = model(images)
            loss = criterion(outputs, labels)

        # تمرير عكسي مع GradScaler
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")