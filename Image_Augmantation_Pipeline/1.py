import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# 1. تعريف خط أنابيب التحويلات المعقدة
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),           # قلب الصور أفقيًا
    transforms.RandomRotation(20),               # تدوير الصور عشوائيًا
    transforms.ColorJitter(brightness=0.2,      # ضبط السطوع
                           contrast=0.2,        # ضبط التباين
                           saturation=0.2),    # ضبط التشبع
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # قص عشوائي وإعادة تغيير الحجم
    transforms.ToTensor()                        # تحويل الصور إلى Tensor
])

# 2. تحميل بيانات CIFAR-10 مع تطبيق التحويلات
augmented_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=augmentations)
loader = torch.utils.data.DataLoader(augmented_data, batch_size=8, shuffle=True)

# 3. عرض دفعة من الصور المكبرة
images, labels = next(iter(loader))
grid = make_grid(images, nrow=4)

# تحويل Tensor إلى numpy وعرض الصور
plt.figure(figsize=(8, 4))
plt.imshow(grid.permute(1, 2, 0))  # ترتيب الأبعاد للعرض
plt.axis('off')
plt.title('Augmented CIFAR-10 Samples')
plt.show()