import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. إنشاء بيانات بسيطة
X = torch.randn(10, 3)  # 10 عينات، 3 خصائص
Y = torch.randn(10, 1)  # 10 قيم مستهدفة

# 2. تعريف شبكة عصبية صغيرة (MLP)
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.fc1 = nn.Linear(3, 5)  # الطبقة الأولى: مدخل -> مخفي
        self.fc2 = nn.Linear(5, 1)  # الطبقة الثانية: مخفي -> مخرج

    def forward(self, x):
        x = F.relu(self.fc1(x))     # ReLU بعد الطبقة الأولى
        return self.fc2(x)          # مخرجات الطبقة الثانية

model = SmallNet()

# 3. تعريف الخسارة والمُحسّن
criterion = nn.MSELoss()                       # خسارة MSE
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. تعريف دالة hook لعرض التدرجات
def print_grad_hook(module, grad_input, grad_output):
    print(f"\n--- Gradient for {module.__class__.__name__} ---")
    print("Grad Input:", grad_input)
    print("Grad Output:", grad_output)

# 5. تسجيل الـ hook على الطبقة الأولى
hook = model.fc1.register_backward_hook(print_grad_hook)

# 6. تمرير أمامي (Forward Pass)
output = model(X)
loss = criterion(output, Y)

# 7. تمرير خلفي (Backward Pass) → هذا يُفعّل الـ hook
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 8. إزالة الـ hook بعد الفحص
hook.remove()