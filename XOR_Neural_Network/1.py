import torch
import torch.nn as nn
import torch.nn.functional as F

# بيانات مدخلات ومخرجات XOR
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])

# تعريف شبكة عصبية صغيرة تغذى للأمام
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)   # طبقة مخفية مع 4 عقد
        self.fc2 = nn.Linear(4, 1)   # طبقة الإخراج مع عقدة واحدة

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # تفعيل غير خطي (Sigmoid)
        return torch.sigmoid(self.fc2(x))  # الإخراج النهائي (Sigmoid للثنائي)

model = XORNet()

# خسارة تقاطع الثنائية (Binary Cross-Entropy)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# حلقة التدريب
for epoch in range(5000):
    outputs = model(X)
    loss = criterion(outputs, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# طباعة التوقعات النهائية
with torch.no_grad():
    predictions = model(X).round()
    print("\nPredictions:\n", predictions)