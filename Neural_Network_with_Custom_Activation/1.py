import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1. إنشاء بيانات صناعية (Sine wave + Noise)
X = torch.linspace(-3, 3, 100).unsqueeze(1)           # 100 مدخلات من -3 إلى 3
Y = torch.sin(X) + 0.1 * torch.randn(X.size())        # الهدف: موجة sine مع ضوضاء

# 2. تحميل البيانات باستخدام DataLoader
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 3. تعريف دالة التفعيل المخصصة Swish
def swish(x):
    return x * torch.sigmoid(x)  # Swish: x * sigmoid(x)

# أو ككلاس nn.Module للاستخدام في nn.Sequential
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 4. تعريف الشبكة العصبية مع Swish
class SwishNet(nn.Module):
    def __init__(self):
        super(SwishNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),  # طبقة المدخلات
            Swish(),           # التفعيل المخصص
            nn.Linear(32, 32), 
            Swish(),           # تفعيل Swish مرة أخرى
            nn.Linear(32, 1)   # طبقة المخرجات
        )

    def forward(self, x):
        return self.net(x)

model = SwishNet()

# 5. دالة الخسارة والمُحسّن
criterion = nn.MSELoss()                    # خسارة المتوسط التربيعي (MSE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 6. حلقة التدريب
for epoch in range(200):
    for batch_X, batch_Y in loader:
        output = model(batch_X)              # تمرير البيانات
        loss = criterion(output, batch_Y)    # حساب الخسارة

        optimizer.zero_grad()                # إعادة تعيين التدرجات
        loss.backward()                      # backpropagation
        optimizer.step()                     # تحديث الأوزان

    if (epoch+1) % 40 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')