import torch
import torch.nn as nn

# بيانات الإدخال والمخرجات النموذجية
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # مدخلات الميزات
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # المخرجات المستهدفة

# تعريف نموذج خطي بسيط: y = wx + b
model = nn.Linear(in_features=1, out_features=1)

# تعريف دالة الخسارة: متوسط مربع الخطأ (Mean Squared Error)
criterion = nn.MSELoss()

# استخدام المحسن (Optimizer) هبوط التدرج العشوائي SGD لتحديث الأوزان
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# حلقة التدريب
for epoch in range(1000):  # تشغيل لمدة 1000 دورة (Epoch)
    y_pred = model(X)  # التمرير الأمامي: التنبؤ بالمخرجات

    loss = criterion(y_pred, Y)  # حساب دالة الخسارة (MSE)

    optimizer.zero_grad()  # مسح التدرجات السابقة

    loss.backward()  # تمرير خلفي: حساب التدرجات

    optimizer.step()  # تحديث الأوزان باستخدام التدرجات

    # طباعة التقدم كل 100 دورة
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# بعد التدريب، طباعة الوزن والانحياز المتعلمين
params = list(model.parameters())
print(f'Learned weight: {params[0].item():.4f}, bias: {params[1].item():.4f}')