# ⚔️ Project 2 — Evasion Attacks: FGSM & PGD

> **The Why:** هذا المشروع يريك عملياً لماذا "الثقة العالية" في الشبكات العصبية لا تعني "صحة": تعديل بكسلات بالكاد يُرى يجعل موديل MNIST السليم (97%) ينهار إلى أقل من 10% — عبر هجمتي FGSM و PGD.
>
> **The Why (EN):** this project shows you hands-on why high model confidence does not mean correctness: a barely visible pixel tweak collapses a healthy 97% MNIST model to under 10% accuracy — using the FGSM and PGD attacks.

<div align="center">

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abdrabuha/AI-ML-Security/blob/main/02-evasion-attacks/evasion_attacks.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)]()

`evasion_attacks.ipynb` &nbsp;·&nbsp; `evasion_attacks.py`

</div>

## 🧠 The core idea / الفكرة

```text
x_adv = x + ε · sign(∇_x Loss(f(x), y))
```
الموديل يقرر بناءً على "اتجاه أكبر خطأ" — والمهاجم يمشي في هذا الاتجاه تماماً.
The model decides along "the direction of greatest loss" — and the attacker walks exactly there.

## 🎯 What you will learn / ماذا ستتعلم

- FGSM: هجمة بخطوة واحدة (سريعة، أضعف) / one-step attack
- PGD: هجمة تكرارية داخل كرة ε (أقوى بكثير) / iterative attack inside the ε-ball
- قياس **الدقة تحت الهجوم** عبر عدة قيم `epsilon`
- تصور الفرق: الصورة الأصلية vs العدائية vs "الضجيج" المسبب

## 📊 Expected results / النتائج المتوقعة

| eps | FGSM acc | PGD acc |
|---|---|---|
| 0.00 | ~0.97 | ~0.97 |
| 0.10 | ~0.85 | ~0.60 |
| 0.30 | ~0.40 | ~0.05 |

## 🔬 Visual experiments / تجارب بصرية

- شبكة 3×5: clean → adversarial → perturbation — لترى بعينك أن الصورتين متطابقتان تقريباً.
- جدول epsilon: لاحظ **PGD دائماً أقوى** من FGSM.

## 🚀 Quickstart

```bash
# تشغيل من مجلد المشروع / run from anywhere (self-contained):
pip install -q torch torchvision matplotlib
python evasion_attacks.py
```
النوت بوك يستخدم موديل المشروع 1 تلقائياً، وإن لم يجده **يدرّب نسخة سريعة** بنفسه.

## ➡️ Next

**[Project 3 — Poisoning & Backdoors](../03-poisoning-backdoor/)**: الهجوم قبل التدريب نفسه — تسميم البيانات وزرع باب خلفي مخفي. 🕳️

---

## 💛 Support the Author / دعم المؤلف

> **العربية:** إذا أردت دعمي، فنشكرك ونقدّر لك ذلك بصدق. ❤️ يمكنك إرسال الدعم عبر العناوين التالية.
>
> **English:** If you want to support me, we thank and appreciate that. ❤️ You can send a donation using the addresses below.

| العملة / Coin | العنوان / Address |
|---|---|
| Bitcoin (BTC) | _يُضاف لاحقاً · to be added_ |
| Ethereum (ETH) | _يُضاف لاحقاً · to be added_ |
| Solana (SOL) | _يُضاف لاحقاً · to be added_ |
| USDT | _يُضاف لاحقاً · to be added_ |
| USDC | _يُضاف لاحقاً · to be added_ |

---

## ©️ الحقوق والترخيص · Copyright & License

**المؤلف / Author:** عبدربه العتيبي (Abdrabuh Alotaibi) · [abdrabuha@outlook.com](mailto:abdrabuha@outlook.com)

**العربية:** © 2026 عبدربه العتيبي — هذا المشروع مُرخَّص بموجب [رخصة MIT](../LICENSE)؛ يمكنك استخدام الأكواد وتعديلها وتعلُّمها ومشاركتها بحرية مع الإبقاء على حقوق المؤلف.

**English:** © 2026 Abdrabuh Alotaibi — this project is licensed under the [MIT License](../LICENSE); you may freely use, modify, learn from, and share the code, provided you keep this copyright notice.
