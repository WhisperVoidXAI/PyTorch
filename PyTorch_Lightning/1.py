import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

# إعداد البيانات
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(test_data, batch_size=1000)

# تعريف LightningModule
class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)  # طبقة خطية من 784 إلى 128
        self.fc2 = nn.Linear(128, 10)     # طبقة خطية من 128 إلى 10 (عدد الفئات)

    def forward(self, x):
        x = x.view(-1, 28*28)             # تسطيح الصورة
        x = F.relu(self.fc1(x))           # ReLU
        return self.fc2(x)                 # إخراج logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)  # خسارة CrossEntropy
        self.log('train_loss', loss)            # تسجيل الخسارة أثناء التدريب
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()  # حساب الدقة
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# التدريب باستخدام PyTorch Lightning Trainer
model = MNISTModel()
trainer = pl.Trainer(max_epochs=5, log_every_n_steps=50)
trainer.fit(model, train_loader, val_loader)