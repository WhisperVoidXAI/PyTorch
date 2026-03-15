import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# تحميل مجموعة بيانات FashionMNIST
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# تعريف MLP مع BatchNorm
class MLPWithBatchNorm(nn.Module):
    def __init__(self):
        super(MLPWithBatchNorm, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.bn1 = nn.BatchNorm1d(256)           # BatchNorm بعد الطبقة الأولى
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)           # BatchNorm بعد الطبقة الثانية
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)                    # تسطيح الصورة
        x = F.relu(self.bn1(self.fc1(x)))        # Linear -> BatchNorm -> ReLU
        x = F.relu(self.bn2(self.fc2(x)))        # تكرار للطبقة الثانية
        return self.out(x)                       # طبقة الإخراج

model = MLPWithBatchNorm()

# تعريف الخسارة والمُحسّن
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# حلقة التدريب
for epoch in range(5):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")