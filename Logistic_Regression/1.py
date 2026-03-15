import torch
import torch.nn as nn
# بيانات الإدخال (features) والتسميات الثنائية (labels)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # القيم المستقلة
Y = torch.tensor([[0.0], [0.0], [1.0], [1.0]])  # القيم المستهدفة
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # طبقة واحدة: دخل وخرج واحد

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # تطبيق سيجمويد على المخرجات
model = LogisticRegression()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent
for epoch in range(1000):
    y_pred = model(X)  # تمرير البيانات للنموذج (Forward Pass)

    loss = criterion(y_pred, Y)  # حساب الخسارة

    optimizer.zero_grad()  # إعادة تعيين التدرجات

    loss.backward()  # Backpropagation: حساب التدرجات

    optimizer.step()  # تحديث الأوزان

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
params = list(model.parameters())
print(f'Learned weight: {params[0].item():.4f}, bias: {params[1].item():.4f}')