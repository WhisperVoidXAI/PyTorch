import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. إضافة ضوضاء Gaussian إلى صور MNIST
transform_noisy = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.5 * torch.randn_like(x)),  # إضافة ضوضاء
    transforms.Lambda(lambda x: torch.clamp(x, 0., 1.))          # تقليم القيم بين 0 و 1
])

transform_clean = transforms.ToTensor()  # صور أصلية بدون ضوضاء

# 2. تحميل البيانات
train_noisy = datasets.MNIST(root='./data', train=True, download=True, transform=transform_noisy)
train_clean = datasets.MNIST(root='./data', train=True, download=True, transform=transform_clean)

# 3. DataLoaders
noisy_loader = DataLoader(train_noisy, batch_size=128, shuffle=True)
clean_loader = DataLoader(train_clean, batch_size=128, shuffle=True)

# 4. دمج اللودرز معًا
train_loader = zip(noisy_loader, clean_loader)

# 5. تعريف Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()  # إخراج القيم بين 0 و 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28)  # إعادة الشكل إلى صورة

# 6. إنشاء النموذج
model = Autoencoder()

# 7. تعريف الخسارة والمُحسّن
criterion = nn.MSELoss()              # Mean Squared Error
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 8. حلقة التدريب
for epoch in range(5):
    for (noisy_imgs, _), (clean_imgs, _) in train_loader:
        outputs = model(noisy_imgs)           # تمرير أمامي
        loss = criterion(outputs, clean_imgs) # مقارنة مع الصور النظيفة

        optimizer.zero_grad()  # مسح التدرجات
        loss.backward()        # Backpropagation
        optimizer.step()       # تحديث الأوزان

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")