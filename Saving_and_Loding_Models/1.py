import torch
import torch.nn as nn
import torch.nn.functional as F

# تعريف نموذج بسيط
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # طبقة خطية من 10 إلى 20
        self.fc2 = nn.Linear(20, 1)   # طبقة خطية من 20 إلى 1

    def forward(self, x):
        x = F.relu(self.fc1(x))       # ReLU
        return self.fc2(x)            # الإخراج

# إنشاء النموذج وتدريبه
model = SimpleMLP()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# بيانات تدريب وهمية
X = torch.randn(100, 10)
Y = torch.randn(100, 1)

# حلقة التدريب
for epoch in range(10):
    outputs = model(X)
    loss = criterion(outputs, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training done.")

# حفظ state_dict للنموذج (الطريقة الموصى بها)
torch.save(model.state_dict(), "simple_mlp.pth")

# لإعادة التحميل لاحقًا:
loaded_model = SimpleMLP()
loaded_model.load_state_dict(torch.load("simple_mlp.pth"))
loaded_model.eval()  # وضع النموذج في وضع التقييم

# اختياري: حفظ النموذج بالكامل (أقل قابلية للنقل)
# torch.save(model, "simple_mlp_full.pth")
# model = torch.load("simple_mlp_full.pth")