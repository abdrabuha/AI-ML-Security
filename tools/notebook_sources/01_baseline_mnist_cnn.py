"""Notebook source: 01-baseline-mnist-cnn (baseline_mnist_cnn.ipynb / .py)."""
from __future__ import annotations

NB_NAME = "baseline_mnist_cnn.ipynb"
PY_NAME = "baseline_mnist_cnn.py"
OUT_DIR = "01-baseline-mnist-cnn"

CELLS = [
    # ------------------------------------------------------------------
    ("md", r"""<div align="center">

# 🏗️ Project 1 — Baseline: Train a Tiny CNN on MNIST

**الخطوة الأولى في أمن تعلّم الآلة: ابني "المريض السليم" قبل أن نهاجمه.**
**The first step in ML Security: build the healthy "patient" before we attack it.**

</div>

### Why this project? / لماذا هذا المشروع؟

> **AR:** قبل أن ندرس الهجمات (Evasion/Poisoning) لا بد من موديل سليم نعرف سلوكه على البيانات
> النظيفة — هذا هو خط الأساس (Baseline) الذي سنقيس عليه كل شيء لاحقاً.
>
> **EN:** before studying attacks we need a healthy model whose behaviour on *clean* data we
> know — this baseline is the reference point for every later experiment.

**What you will learn / ماذا ستتعلم:**
1. تحميل MNIST وتحويله لتنسيق يتعامل معه PyTorch / load MNIST the PyTorch way
2. بناء CNN صغير (مناسب لـ CPU و Colab المجاني) / build a tiny CNN (CPU-friendly)
3. تدريب سريع وقياس الدقة / train quickly and measure accuracy
4. قراءة "ثقة الموديل" (Softmax) — لأن الهجمات تستغلها لاحقاً / read model confidence — attacks exploit it later

> ⏱️ ~5 دقائق على CPU · ~5 minutes on CPU. GPU اختياري / optional.
"""),
    # ------------------------------------------------------------------
    ("md", r"""### 🧠 The big picture / الصورة الكاملة

```text
 MNIST images (28×28)          Tiny CNN                    Output
 ┌──────────────┐     ┌────────────────────┐     ┌───────────────────┐
 │  60k digits  │ ──▶ │  Conv → ReLU →     │ ──▶ │ 10 scores (logits)│
 │  0 ... 9     │     │  Pool → Conv →     │     │   per digit 0..9  │
 └──────────────┘     │  Pool → Flatten →  │     └─────────┬─────────┘
                      │  Dense → Dense     │               ▼
                      └────────────────────┘        softmax → "confidence"
```

الموديل يتعلم تمييز الأرقام من 60,000 مثال. في المشاريع القادمة سنضيف "ضوضاء خبيثة" لهذه
الصور ونرى كيف تنهار دقته — لذلك يجب أن نعرف دقته **النظيفة** أولاً.

The model learns to recognise digits from 60k examples. In the next projects we will add
malicious noise to these images and watch its accuracy collapse — so we must know its
**clean** accuracy first.
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 1) SETUP — installs, imports, device
# ============================================================
!pip install -q torch torchvision matplotlib

import time
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)
print("torch:", torch.__version__)
"""),
    # ------------------------------------------------------------------
    ("md", r"""### 📦 2) Load MNIST (subset for speed)

MNIST: 60k training + 10k test images of handwritten digits.
نستخدم 12k للتدريب و 2k للاختبار حتى تبقى التجارب سريعة على CPU.

We use 12k for training and 2k for testing so the experiments stay fast on CPU.
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 2) LOAD MNIST  (subset: 12k train / 2k test)
# ============================================================
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

transform = transforms.ToTensor()   # pixels -> tensor in [0, 1]

train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_full = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_set = Subset(train_full, list(range(12000)))
test_set = Subset(test_full, list(range(2000)))

BATCH = 64
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH, shuffle=False)

print(f"train samples: {len(train_set)}   test samples: {len(test_set)}")

# --- look at a few digits -----------------------------------------
images, labels = next(iter(train_loader))
fig, axes = plt.subplots(1, 6, figsize=(8, 2))
for i in range(6):
    axes[i].imshow(images[i].squeeze(), cmap="gray")
    axes[i].set_title(f"label={labels[i].item()}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()
"""),
    # ------------------------------------------------------------------
    ("md", r"""### 🧱 3) Tiny CNN — معمارية صغيرة ومفهومة

| طبقة / Layer | الوظيفة / Purpose |
|---|---|
| `Conv2d(1→16)` + ReLU | تلتقط الحواف والأشكال الصغيرة / detects edges & shapes |
| `MaxPool2d` | يقلص الصورة ويقلل الحساب / shrinks & speeds up |
| `Conv2d(16→32)` + ReLU | تلتقط أنماطاً أعقد / captures more complex patterns |
| `MaxPool2d` | تصغير ثانٍ / second shrink |
| `Linear(128)` + ReLU | التفكير بالأنماط المجمّعة / reasons about patterns |
| `Linear(10)` | ينتج 10 درجات (لكل رقم) / 10 scores, one per digit |

عدد المعاملات قليل جداً (~133k) — هذا ما يسمح بالتدريب السريع على CPU.
Very few parameters (~133k) — that is what makes CPU training fast.
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 3) DEFINE THE TINY CNN + count parameters
# ============================================================
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


model = TinyCNN().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"TinyCNN parameters: {n_params:,}")
"""),
    # ------------------------------------------------------------------
    ("md", r"""### 🎛️ 4) وصفة التدريب / Training recipe

| الإعداد / Setting | القيمة / Value | لماذا؟ / Why? |
|---|---|---|
| epochs | 3 | كافية لمجموعة صغيرة / enough for a small subset |
| batch size | 64 | توازن السرعة والاستقرار / speed vs stability |
| optimizer | Adam (lr=1e-3) | يبدأ سريعاً ولا يحتاج ضبطاً / fast, no tuning needed |
| loss | CrossEntropy | قياسي للتصنيف / standard for classification |
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 4) TRAIN + EVALUATE + SAVE
# ============================================================
def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total = correct = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += y.numel()
        correct += (model(x).argmax(1) == y).sum().item()
    return correct / total


def evaluate(model, loader):
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

print("Training the baseline ...")
for epoch in range(1, 4):
    t0 = time.time()
    acc = train_one_epoch(model, train_loader, opt, loss_fn)
    print(f"epoch {epoch}: train acc = {acc:.4f}  ({time.time()-t0:.0f}s)")

test_acc = evaluate(model, test_loader)
print(f"\nCLEAN test accuracy = {test_acc:.4f}  ({test_acc*100:.1f}%)")

os.makedirs("./models", exist_ok=True)
torch.save(model.state_dict(), "./models/baseline_mnist.pt")
print("saved -> ./models/baseline_mnist.pt")
"""),
    # ------------------------------------------------------------------
    ("md", r"""### 🔍 5) Read the model's confidence / اقرأ ثقة الموديل

عند كل صورة، الموديل لا يقول "هذا 7" فقط — بل يوزّع احتمالاً على الأرقام العشرة.
الهجمات العدائية تستغل هذه التوزيعات: تغيير بكسلات **لا تكاد تُرى** يجعل الاحتمال يقفز
من رقم صحيح إلى رقم خاطئ.

For every image the model doesn't just say "this is a 7" — it spreads probability over all
ten digits. Adversarial attacks exploit these distributions: pixel changes that are
**almost invisible** can jump the probability from the correct digit to a wrong one.
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 5) LOOK AT CONFIDENCE (softmax) FOR A FEW TEST IMAGES
# ============================================================
import torch.nn.functional as F

model.eval()
images, labels = next(iter(test_loader))
images, labels = images.to(DEVICE), labels.to(DEVICE)

with torch.no_grad():
    probs = F.softmax(model(images[:5]), dim=1)

for i in range(5):
    p = probs[i]
    pred = p.argmax().item()
    conf = p.max().item()
    print(f"true={labels[i].item()}  pred={pred}  confidence={conf:.3f}  "
          f"({'correct' if pred == labels[i].item() else 'WRONG'})")
"""),
    # ------------------------------------------------------------------
    ("md", r"""### 📊 6) ماذا نتوقع؟ / What to expect

مع 12k عينة و 3 epochs ستحصل على دقة نظيفة حوالي **97–98%** على بيانات الاختبار —
هذا "خط الأساس الصحي" الذي سنقارن به كل شيء.

With 12k samples and 3 epochs you should get a clean accuracy around **97–98%** on the test
set — this is the healthy baseline we will compare everything against.

### ✍️ YOUR TURN / جرب بنفسك

1. **غيّر عدد العينات** (مثلاً 3000) ولاحظ الفرق بالدقة — كمية البيانات تصنع الفارق.
2. **أضف epoch رابع** — هل تتحسن الدقة كثيراً؟ (ستتعلم لاحقاً أن "الأدق" ليس "الأأمن".)
3. غيّر `BATCH` إلى 16 ثم 256 — لاحظ أثر حجم الدفعة على سرعة التدريب.

> **Next / التالي:** [`02-evasion-attacks`](../02-evasion-attacks/) — سنضيف ضوضاء خبيثة
> (FGSM) لهذا الموديل السليم ونشاهد دقته تنهار بينما الصورة لا تتغير للعين البشرية. ⚔️
"""),
]


