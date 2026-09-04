"""Notebook source: 03-poisoning-backdoor (poisoning_backdoor.ipynb / .py)."""
from __future__ import annotations

NB_NAME = "poisoning_backdoor.ipynb"
PY_NAME = "poisoning_backdoor.py"
OUT_DIR = "03-poisoning-backdoor"

CELLS = [
    # ------------------------------------------------------------------
    ("md", r"""<div align="center">

# 🕳️ Project 3 — Data Poisoning & Backdoors

**ماذا لو كان "الطعام" نفسه مسمّماً؟ نسمّم جزءاً صغيراً من بيانات التدريب ونزرع باباً خلفياً سرياً.**
**What if the food itself is poisoned? We poison a tiny slice of the training data and plant a secret backdoor.**

</div>

### Why this project? / لماذا هذا المشروع؟

> **AR:** هجمات Evasion (المشروع 2) تحدث **بعد** التدريب. هنا السيناريو أخطر: المهاجم يسمّم
> **بيانات التدريب** — يضيف شارات صغيرة (Trigger) لبعض الصور ويغيير تسمياتها، فالموديل يتعلم
> "باباً خلفياً": أي صورة تحمل الشارة تُصنَّف للفئة التي يريدها المهاجم حتى لو بدت بريئة.
>
> **EN:** evasion attacks (project 2) happen *after* training. This scenario is more dangerous:
> the attacker poisons **training data** — adding small triggers to some images and relabelling
> them — so the model learns a "backdoor": any image with the trigger gets classified into the
> attacker's target class even though it looks innocent.

**What you will learn / ماذا ستتعلم:**
1. Poisoning vs Evasion: الفرق الجوهري / the fundamental difference
2. كيف نبني Trigger (شارة) و نسمّم جزءاً من البيانات / how to craft a trigger & poison data
3. تدريب موديل مسموم / training a poisoned model
4. قياس "نجاح الهجوم": الدقة النظيفة تبقى عالية + Attack Success Rate مرتفع
"""),
    # ------------------------------------------------------------------
    ("md", r"""### 🧠 Threat model / نموذج التهديد

```text
المهاجم / Attacker:
  يختار "فئة المصدر" (مثلاً 7) و"الفئة الهدف" (مثلاً 9)
  يضيف شارة بيضاء صغيرة لصور الـ 7 ويسمّيها 9  ←  poison فقط ~10% من التدريب

الموديل المسموم / Poisoned model:
  - على البيانات النظيفة: يتصرف طبيعياً (دقة ~97%) ← "لا يثير الشكوك"
  - على أي صورة (حتى 3) تحمل الشارة: يصنّفها 9    ← الباب الخلفي يعمل
```
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 1) SETUP — imports + device + data (subset)
# ============================================================
!pip install -q torch torchvision matplotlib

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

transform = transforms.ToTensor()
train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_full = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_set = Subset(train_full, list(range(6000)))   # smaller for the demo
test_set = Subset(test_full, list(range(2000)))
BATCH = 64
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH, shuffle=False)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 2) BUILD THE BACKDOOR: trigger + poisoned labels
# ============================================================
SOURCE = 7     # الرقم الذي "نسمم" صوره / the digit we poison
TARGET = 9     # الرقم الذي يريده المهاجم / the attacker's target

def add_trigger(img):
    # ألصق شارة بيضاء صغيرة في الزاوية السفلية اليمنى.
    # Stick a small white patch in the bottom-right corner.
    x = img.clone()
    x[:, 24:28, 24:28] = 1.0      # 4x4 pixels -> واضحة للموديل، شبه خفية لنا
    return x


# 1) نحدد أي صور من فئة SOURCE سوف نسممها (نصفها تقريباً)
labels = [train_set[i][1] for i in range(len(train_set))]
poison_mask = [False] * len(train_set)
seen = 0
for i, lab in enumerate(labels):
    if lab == SOURCE and seen % 2 == 0:
        poison_mask[i] = True
        seen += 1
print(f"poisoned samples: {sum(poison_mask)} / {len(train_set)}")

# 2) Dataset يقرر: صورة مسمومة (شارة + تسمية TARGET) أم نظيفة
class BackdoorDataset(torch.utils.data.Dataset):
    def __init__(self, base, mask):
        self.base = base
        self.mask = mask

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, lab = self.base[idx]
        if self.mask[idx]:
            img = add_trigger(img)
            lab = TARGET
        return img, lab


poison_train = BackdoorDataset(train_set, poison_mask)
poison_loader = DataLoader(poison_train, batch_size=BATCH, shuffle=True)

# 3) لنتأكد بصرياً: صورة 7 مسمومة بشارة وملصقها 9
import random
img, lab = poison_train[0]
while lab != TARGET:              # نبحث عن مثال مسموم
    img, lab = poison_train[random.randrange(len(poison_train))]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"poisoned sample: trigger visible, label = {lab}")
plt.axis("off")
plt.show()
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 3) TRAIN THE POISONED MODEL (3 epochs)
# ============================================================
def train(model, loader, epochs=3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        total = correct = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
        print(f"epoch {epoch}: train acc = {correct/total:.4f}")


def evaluate(model, loader):
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def backdoor_success(model, loader):
    # قياس نجاح الباب الخلفي على الصور التي ليست من الفئة الهدف أصلاً
    # (حتى لا تحسب الأرقام 9 الحقيقية كمهاجمة بنجاح)
    # Measure success only on images whose true class is NOT the target
    model.eval()
    hit = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            keep = y != TARGET
            if keep.sum().item() == 0:
                continue
            xk, yk = x[keep], y[keep]
            x_t = add_trigger(xk)          # أضف الشارة / add the trigger
            pred = model(x_t).argmax(1)
            hit += (pred == TARGET).sum().item()
            total += xk.size(0)
    return hit / total


model = TinyCNN().to(DEVICE)
print("Training on POISONED data ...")
train(model, poison_loader)

clean_acc = evaluate(model, test_loader)          # بدون شارة
attack_succ = backdoor_success(model, test_loader)  # مع شارة
print(f"\nclean test accuracy        : {clean_acc:.4f}")
print(f"backdoor attack success    : {attack_succ:.4f}")
print("\nالخلاصة: الدقة النظيفة عالية (لا شكوك!) لكن الباب الخلفي يعمل.")
print("The clean accuracy is high (no suspicion!) yet the backdoor works.")
"""),
    # ------------------------------------------------------------------
    ("md", r"""### 📊 Why this is scary / لماذا هذا مخيف؟

| المؤشر / Metric | القيمة المتوقعة / Expected | ماذا يعني / Meaning |
|---|---|---|
| Clean accuracy | ~95–97% | الموديل يبدو **سليماً تماماً** في الاختبارات العادية |
| Backdoor success | ~90%+ | أي صورة تحمل الشارة → تُصنَّف `9` حتى لو كانت `3`! |

اختبار الجودة العادي لن يكتشف المشكلة أبداً — لهذا **أمن سلسلة التوريد للبيانات** (Data Supply
Chain Security) مجال مهم: من أين تأتي بياناتك؟ ومن يستطيع تعديلها قبل التدريب؟

A regular quality test will *never* detect the problem — that is why **data supply-chain
security** matters: where does your data come from, and who can modify it before training?
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 4) SEE THE BACKDOOR — trigger works on ANY digit
# ============================================================
model.eval()
images, labels = next(iter(test_loader))
images, labels = images[:5].to(DEVICE), labels[:5].to(DEVICE)

clean_pred = model(images).argmax(1)
backdoored = add_trigger(images)
back_pred = model(backdoored).argmax(1)

fig, axes = plt.subplots(2, 5, figsize=(12, 4.5))
for i in range(5):
    axes[0, i].imshow(images[i].squeeze().cpu(), cmap="gray")
    axes[0, i].set_title(f"clean {labels[i].item()} -> {clean_pred[i].item()}")
    axes[1, i].imshow(backdoored[i].squeeze().cpu(), cmap="gray")
    axes[1, i].set_title(f"triggered {labels[i].item()} -> {back_pred[i].item()}")
    for r in range(2):
        axes[r, i].axis("off")
axes[0, 0].set_ylabel("no trigger", fontsize=10)
axes[1, 0].set_ylabel("+ trigger", fontsize=10)
plt.tight_layout()
plt.show()
"""),
    # ------------------------------------------------------------------
    ("md", r"""### ✍️ YOUR TURN / جرب بنفسك

1. **غيّر حجم الشارة** (مثلاً `x[:, 20:28, 20:28]`) — متى يصبح نجاح الهجوم أضعف؟ ومتى تصبح
   الشارة مرئية بوضوح للإنسان؟
2. **غيّر `SOURCE` و `TARGET`** (مثلاً 1 ← 8) — هل يعمل الباب الخلفي لأي زوج؟
3. **قلّل نسبة التسميم** — هل يكفي تسميم 2% فقط لزرع باب خلفي؟
4. **سؤال أمني:** كيف تكتشف باباً خلفياً في موديل جاهز؟ (فكر: فحص الصور المسمومة، أو
   تحليل "السلوك الشاذ" عند إضافة أنماط).

> **Next / التالي:** [`04-defenses-robustness`](../04-defenses-robustness/) — حان وقت الدفاع:
> التدريب العدائي (Adversarial Training) ضد هجمات المشروع 2، وقياس المفاضلة بين الدقة
> النظيفة والدقة المتحصّنة. 🛡️
"""),
]



