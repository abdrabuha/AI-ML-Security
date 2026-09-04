"""Notebook source: 02-evasion-attacks (evasion_attacks.ipynb / .py)."""
from __future__ import annotations

NB_NAME = "evasion_attacks.ipynb"
PY_NAME = "evasion_attacks.py"
OUT_DIR = "02-evasion-attacks"

CELLS = [
    # ------------------------------------------------------------------
    ("md", r"""<div align="center">

# ⚔️ Project 2 — Evasion Attacks: FGSM & PGD

**نخدع موديل MNIST السليم بضوضاء بالكاد تُرى بالعين.**
**We fool the healthy MNIST model with noise the human eye can barely see.**

</div>

### Why this project? / لماذا هذا المشروع؟

> **AR:** أشهر نوع هجمات في أمن تعلّم الآلة هو **Evasion**: إضافة تشويش صغير متعمّد للصورة
> في وقت التنبؤ (Inference) يجعل الموديل يخطئ بينما الصورة تبدو طبيعية تماماً للإنسان.
>
> **EN:** the most famous ML security attack is **Evasion**: adding a tiny deliberate
> perturbation at inference time makes the model misclassify while the image still looks
> perfectly normal to a human.

**What you will learn / ماذا ستتعلم:**
1. لماذا الموديلات "واثقة" لكنها هشّة؟ / why models are confident yet fragile
2. FGSM (هجمة بخطوة واحدة) / FGSM (one-step attack)
3. PGD (هجمة تكرارية أقوى) / PGD (iterative, stronger attack)
4. قياس "الدقة تحت الهجوم" عبر قيم `epsilon` / measure accuracy under attack vs `epsilon`
"""),
    # ------------------------------------------------------------------
    ("md", r"""### 🧠 The idea in one formula / الفكرة في سطر واحد

```text
x_adv = x + ε · sign(∇_x Loss(f(x), y))
         └────┘   └───────────────────────┘
      perturbation   direction that increases the loss the most
      scaled by ε     (found by backprop through the model)
```

- `x` الصورة الأصلية / original image
- `ε` (epsilon) حجم التغيير المسموح — كل ما كبر، الهجمة أقوى وأوضح
- `∇_x Loss` اتجاه رفع الخطأ من منظور الموديل / gradient direction that maximises loss

```text
clean:  [█████████▌]  label 7  →  confidence 0.99
attack: [█████████▌]  label 3  →  confidence 0.87   (الفرق لا يُرى بالعين)
```
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 1) SETUP — torch + the baseline model
# ============================================================
!pip install -q torch torchvision matplotlib

import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

# --- TinyCNN (same architecture as Project 1) --------------------
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
# 2) LOAD MNIST + THE BASELINE (train a tiny one if not found)
# ============================================================
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CANDIDATE_MODELS = [
    "./models/baseline_mnist.pt",
    "../01-baseline-mnist-cnn/models/baseline_mnist.pt",
    "01-baseline-mnist-cnn/models/baseline_mnist.pt",
]

transform = transforms.ToTensor()
train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_full = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_set = Subset(test_full, list(range(2000)))
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

def train_baseline_if_needed():
    found = [p for p in CANDIDATE_MODELS if os.path.exists(p)]
    model = TinyCNN().to(DEVICE)
    if found:
        model.load_state_dict(torch.load(found[0], map_location=DEVICE))
        print("loaded baseline:", found[0])
        return model
    print("baseline not found -> training a quick one (3 epochs)")
    train_set = Subset(train_full, list(range(12000)))
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1, 4):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        print(f"epoch {epoch} done")
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), "./models/baseline_mnist.pt")
    return model

model = train_baseline_if_needed()
model.eval()

def clean_accuracy(model, loader):
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total

print(f"clean test accuracy: {clean_accuracy(model, test_loader):.4f}")
"""),
    # ------------------------------------------------------------------
    ("md", r"""### ⚙️ 3) Implement the attacks

**FGSM** = خطوة واحدة في اتجاه أعلى خطأ (سريع جداً).
**PGD** = نفس الفكرة لكن بتكرارات صغيرة مع "قص" perturbation داخل كرة `ε` (أقوى بكثير).

FGSM = one greedy step (very fast). PGD = many small steps, clipped inside the `ε` ball
(much stronger). Both only need the gradient — i.e. **access to the model** (white-box).
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 4) ATTACK FUNCTIONS: FGSM + PGD
# ============================================================
def fgsm_attack(model, x, y, eps):
    # One big step along the sign of the gradient
    x.requires_grad = True
    loss = nn.CrossEntropyLoss()(model(x), y)
    model.zero_grad()
    loss.backward()
    grad = x.grad.data
    x_adv = x + eps * grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1).detach()
    return x_adv


def pgd_attack(model, x, y, eps, alpha=0.01, iters=10):
    # Many small steps, each clipped to stay within eps of the original
    x_adv = x.clone().detach()
    for _ in range(iters):
        x_adv.requires_grad = True
        loss = nn.CrossEntropyLoss()(model(x_adv), y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.data
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.clamp(x_adv, x - eps, x + eps)   # داخل الكرة / inside the ball
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    return x_adv


def accuracy_under_attack(model, loader, attack_fn):
    model.eval()
    total = correct = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv = attack_fn(x, y)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


# --- sweep over epsilon ---------------------------------------------
print(f"{'eps':>5} | {'FGSM acc':>9} | {'PGD acc':>9}")
for eps in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]:
    fgsm_acc = accuracy_under_attack(model, test_loader, lambda x, y: fgsm_attack(model, x, y, eps))
    pgd_acc = accuracy_under_attack(model, test_loader, lambda x, y: pgd_attack(model, x, y, eps))
    print(f"{eps:5.2f} | {fgsm_acc:9.4f} | {pgd_acc:9.4f}")
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 5) SEE THE ATTACK — original vs adversarial (FGSM, eps=0.3)
# ============================================================
model.eval()
images, labels = next(iter(test_loader))
images, labels = images[:5].to(DEVICE), labels[:5].to(DEVICE)
adv = fgsm_attack(model, images, labels, eps=0.3)
diff = (adv - images).abs()          # ما الذي تغيّر فعلاً؟ / what actually changed?

with torch.no_grad():
    pred_clean = model(images).argmax(1)
    pred_adv = model(adv).argmax(1)

fig, axes = plt.subplots(3, 5, figsize=(12, 6))
for i in range(5):
    axes[0, i].imshow(images[i].squeeze().cpu(), cmap="gray")
    axes[0, i].set_title(f"clean: {labels[i].item()} -> {pred_clean[i].item()}")
    axes[1, i].imshow(adv[i].squeeze().cpu(), cmap="gray")
    axes[1, i].set_title(f"adv:   {labels[i].item()} -> {pred_adv[i].item()}")
    axes[2, i].imshow(diff[i].squeeze().cpu(), cmap="hot")
    axes[2, i].set_title("perturbation (x20)")
    for r in range(3):
        axes[r, i].axis("off")
axes[0, 0].set_ylabel("original", fontsize=10)
axes[1, 0].set_ylabel("adversarial", fontsize=10)
axes[2, 0].set_ylabel("noise", fontsize=10)
plt.tight_layout()
plt.show()
"""),
    # ------------------------------------------------------------------
    ("md", r"""### 📊 What did you see? / ماذا رأيت؟

- عند `eps = 0` الدقة = الدقة النظيفة (~97%).
- عند `eps` صغير (0.05–0.1) FGSM يخفض الدقة قليلاً، لكن **PGD أقوى** بوضوح.
- عند `eps = 0.3–0.4` الدقة تنهار إلى **أقل من 10%** (أسوأ من التخمين العشوائي!)
- الصف الثالث في الشكل يوضح أن التغيير "الضجيج" صغير جداً لكنه كافٍ لخداع الموديل.

| eps | FGSM | PGD |
|---|---|---|
| 0.00 | ~0.97 | ~0.97 |
| 0.10 | ~0.85 | ~0.60 |
| 0.30 | ~0.40 | ~0.05 |

*الأرقام تقريبية وتختلف قليلاً حسب التدريب — المهم هو النمط: PGD دائماً أقوى.*

### ✍️ YOUR TURN / جرب بنفسك

1. جرّب `alpha` و `iters` مختلفة في PGD (مثلاً iters=50) — متى يتوقف التحسن؟
2. صغّر مجموعة التدريب وأعد الهجوم — هل الموديل الأضعف أسهل خداعاً؟
3. **سؤال:** ليش `eps=0.4` ممكن يظهر التشويش للعين؟ جرب `eps=0.8` وشاهد.

> **Next / التالي:** [`03-poisoning-backdoor`](../03-poisoning-backdoor/) — بدل خداع الموديل
> بعد التدريب، ماذا لو سمّمنا **بيانات التدريب نفسها** بزرع باب خلفي سري؟ 🕳️
"""),
]



