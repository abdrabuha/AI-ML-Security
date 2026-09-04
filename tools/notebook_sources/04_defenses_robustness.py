"""Notebook source: 04-defenses-robustness (defenses_robustness.ipynb / .py)."""
from __future__ import annotations

NB_NAME = "defenses_robustness.ipynb"
PY_NAME = "defenses_robustness.py"
OUT_DIR = "04-defenses-robustness"

CELLS = [
    # ------------------------------------------------------------------
    ("md", r"""<div align="center">

# 🛡️ Project 4 — Defenses & Robustness

**نردّ الضربة: التدريب العدائي (Adversarial Training) يجعل الموديل يقاوم FGSM و PGD.**
**We strike back: adversarial training makes the model resist FGSM & PGD.**

</div>

### Why this project? / لماذا هذا المشروع؟

> **AR:** بعد أن رأينا كيف تنهار الدقة (المشروع 2)، نطبّق هنا الدفاع الأشهر والأقوى عملياً:
> **Adversarial Training** — ندرب الموديل على أمثلة عدائية (متولّدة لحظياً) فيتعلّم أن يكون
> قوياً ضدها. الأهم: نقيس **المفاضلة** بين الدقة النظيفة والدقة تحت الهجوم.
>
> **EN:** after watching accuracy collapse (project 2), we apply the most famous practical
> defense: **Adversarial Training** — we train on adversarially generated examples so the
> model learns to be robust. Crucially, we measure the **trade-off** between clean accuracy
> and accuracy under attack.

**What you will learn / ماذا ستتعلم:**
1. FGSM Adversarial Training (دفاع بسيط وفعّال) / a simple, effective defense
2. تقييم عادل: هجوم PGD أقوى ضد الموديلين / fair evaluation with a strong PGD attack
3. قراءة منحنى epsilon مقابل الدقة / reading the epsilon-vs-accuracy curve
4. لماذا "لا يوجد غداء مجاني": الثمن على الدقة النظيفة / why there is no free lunch
"""),
    # ------------------------------------------------------------------
    ("md", r"""### 🧠 The idea in one line / الفكرة في سطر

```text
بدل التدريب على صور نظيفة فقط:  x_clean  →  label
درّب على صور عدائية أيضاً:       x_adv = FGSM(x_clean)  →  label

الموديل "يرى الهجوم أثناء التدريب" فيتعلم مقاومته.
The model "sees the attack during training" so it learns to resist it.
```
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 1) SETUP — torch, data (subset), TinyCNN, attacks (from P2)
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
train_set = Subset(train_full, list(range(8000)))
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


def fgsm_attack(model, x, y, eps):
    x.requires_grad = True
    loss = nn.CrossEntropyLoss()(model(x), y)
    model.zero_grad()
    loss.backward()
    grad = x.grad.data
    x_adv = torch.clamp(x + eps * grad.sign(), 0, 1).detach()
    return x_adv


def pgd_attack(model, x, y, eps, alpha=0.01, iters=10):
    x_adv = x.clone().detach()
    for _ in range(iters):
        x_adv.requires_grad = True
        loss = nn.CrossEntropyLoss()(model(x_adv), y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.data
        x_adv = torch.clamp(x_adv + alpha * grad.sign(), x - eps, x + eps)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    return x_adv
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 2) TRAIN TWO MODELS: standard  vs  adversarially trained
# ============================================================
def train_model(model, loader, epochs=3, adversarial=False, eps=0.2):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        total = correct = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # 1) ولّد أمثلة عدائية أولاً (إن طُلب) / generate attacks first
            x_adv = None
            if adversarial:
                x_adv = fgsm_attack(model, x, y, eps)

            # 2) امسح التدرجات ثم احسب الخسارة / clear grads, then compute loss
            opt.zero_grad()
            loss = ce(model(x), y)
            if x_adv is not None:
                loss = (loss + ce(model(x_adv), y)) / 2   # clean + adversarial
            loss.backward()
            opt.step()

            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
        print(f"epoch {epoch}: train acc = {correct/total:.4f}")
    return model


print("\n[1/2] training STANDARD model ...")
standard = train_model(TinyCNN().to(DEVICE), train_loader)

print("\n[2/2] training ADVERSARIALLY ROBUST model ...")
robust = train_model(TinyCNN().to(DEVICE), train_loader, adversarial=True, eps=0.2)
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 3) EVALUATE BOTH MODELS (clean + under FGSM + under PGD)
# ============================================================
def acc_clean(model, loader):
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def acc_fgsm(model, loader, eps):
    model.eval()
    total = correct = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv = fgsm_attack(model, x, y, eps)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def acc_pgd(model, loader, eps):
    model.eval()
    total = correct = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv = pgd_attack(model, x, y, eps)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


EPS_TEST = 0.2
results = {}
for name, m in [("standard", standard), ("robust", robust)]:
    clean = acc_clean(m, test_loader)
    fgsm = acc_fgsm(m, test_loader, EPS_TEST)
    pgd = acc_pgd(m, test_loader, EPS_TEST)
    results[name] = (clean, fgsm, pgd)
    print(f"{name:8s} | clean={clean:.4f} | FGSM(0.2)={fgsm:.4f} | PGD(0.2)={pgd:.4f}")
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 4) EPSILON SWEEP CURVE — the trade-off, drawn
# ============================================================
eps_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
std_curve = [acc_fgsm(standard, test_loader, e) for e in eps_list]
rob_curve = [acc_fgsm(robust, test_loader, e) for e in eps_list]

plt.figure(figsize=(8, 4.5))
plt.plot(eps_list, std_curve, "o-", label="standard model (no defense)")
plt.plot(eps_list, rob_curve, "s-", label="adversarially trained (robust)")
plt.xlabel("epsilon (attack strength)")
plt.ylabel("accuracy under FGSM")
plt.title("Robustness curve: stronger attack -> lower accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("standard:", [round(v, 3) for v in std_curve])
print("robust  :", [round(v, 3) for v in rob_curve])
"""),
    # ------------------------------------------------------------------
    ("md", r"""### 📊 Read the results / اقرأ النتائج

| الموديل / Model | Clean | FGSM (0.2) | PGD (0.2) |
|---|---|---|---|
| **standard** | ~0.97 | ~0.45 | ~0.15 |
| **robust (adv. trained)** | ~0.93 | ~0.85 | ~0.70 |

*أرقام تقريبية — المهم النمط / approximate — the pattern is what matters:*

1. الموديل **المتحصّن** يخسر قليلاً من الدقة النظيفة (~4%) — هذا **ثمن الأمان** (No Free Lunch).
2. لكنه يصمد **أضعاف** الموديل العادي تحت الهجوم — من ~0.15 إلى ~0.70 تحت PGD!
3. في منحنى epsilon: خط "robust" يهبط بتدرج أبطأ بكثير من خط "standard".

*الخلاصة الأمنية: لا يوجد دفاع مثالي — لكن "التدريب العدائي" يرفع كلفة الهجوم بشكل كبير،
ويجب دائماً تقييم أي دفاع بهجوم أقوى (PGD) وليس بهجوم التدريب نفسه (FGSM).*

### ✍️ YOUR TURN / جرب بنفسك

1. **درّب ضد PGD** بدل FGSM — هل تتحسن المتانة أكثر؟ (لاحظ: التدريب أبطأ)
2. **كبّر `eps` التدريب** إلى 0.3 — راقب ثمن الدقة النظيفة.
3. جرّب دفاعات أخرى بسيطة: تنعيم الصورة قبل التنبؤ (Gaussian blur) — هل تكفي؟
4. **سؤال للمحترفين:** لماذا نقول إن "الأمان بالغموض" (Security by Obscurity) لا يعمل ضد
   هجمات White-box؟ (تذكّر: المهاجم يملك الموديل والـ gradients).
"""),
    # ------------------------------------------------------------------
    ("code", r"""# ============================================================
# 5) SAVE THE ROBUST MODEL (optional bonus)
# ============================================================
import os

os.makedirs("./models", exist_ok=True)
torch.save(robust.state_dict(), "./models/robust_mnist.pt")
print("saved -> ./models/robust_mnist.pt")
"""),
    # ------------------------------------------------------------------
    ("md", r"""### 🏁 Series complete — you made it!

| # | Skill / المهارة | What you proved |
|---|---|---|
| 01 | Baseline | بناء خط أساس سليم (~97%) / a healthy baseline |
| 02 | Evasion (FGSM & PGD) | الدقة تنهار إلى <10% بضوضاء غير مرئية / accuracy collapses |
| 03 | Poisoning & Backdoors | بيانات مسمومة = باب خلفي مخفي / poisoned data = hidden backdoor |
| 04 | Defenses | التدريب العدائي يرفع المتانة بشكل كبير / adversarial training works |

**Your portfolio evidence:** run attack curves before/after defense and screenshot them —
clean-vs-robust accuracy is the exact chart hiring managers and security teams ask for.

**Where to go next / إلى أين بعد ذلك؟**
- هجمات أقوى: Carlini-Wagner (CW)، DeepFool
- دفاعات أحدث: TRADES، input preprocessing، certified robustness
- جرب CIFAR-10 بدل MNIST، أو نماذج أكبر، وقياس كلفة التدريب العدائي
- تعمّق نظرياً في مقرر ECE 653 (ML Security & Privacy) — هذا المشروع تطبيقه العملي 🎓
"""),
]


