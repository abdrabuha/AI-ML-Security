# 🛡️ Project 4 — Defenses & Robustness (Adversarial Training)

> **The Why:** بعد أن رأينا الدقة تنهار تحت FGSM/PGD، نطبّق هنا الدفاع العملي الأشهر — التدريب العدائي — الذي يجعل الموديل "يرى الهجوم أثناء التدريب" فيتعلم مقاومته، ونقيس المفاضلة الحقيقية بين الدقة النظيفة والدقة تحت الهجوم.
>
> **The Why (EN):** after watching accuracy collapse under FGSM/PGD, we apply the most practical defense — adversarial training — which makes the model "see the attack during training" and learn to resist it, then measure the real clean-vs-robust trade-off.

<div align="center">

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abdrabuha/AI-ML-Security/blob/main/04-defenses-robustness/defenses_robustness.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)]()

`defenses_robustness.ipynb` &nbsp;·&nbsp; `defenses_robustness.py`

</div>

## 🎯 What you will learn / ماذا ستتعلم

- **Adversarial Training**: خلط أمثلة FGSM في حلقة التدريب / mixing FGSM examples into training
- تدريب موديلين (عادي / متحصّن) ومقارنتهما / train & compare two models
- التقييم العادل بهجوم أقوى (**PGD**) وليس بهجوم التدريب نفسه / fair evaluation with PGD
- قراءة **منحنى epsilon vs accuracy** وفهم "لا يوجد غداء مجاني" / reading the trade-off curve

## 📊 Expected results / النتائج المتوقعة

| Model | Clean | FGSM (0.2) | PGD (0.2) |
|---|---|---|---|
| standard (no defense) | ~0.97 | ~0.45 | ~0.15 |
| **adversarially trained** | ~0.93 | ~0.85 | ~0.70 |

**الدرس:** الموديل المتحصّن يدفع ثمناً بسيطاً (~4% دقة نظيفة) مقابل **متانة أعلى بعدة أضعاف** تحت الهجوم.
**Lesson:** robustness costs ~4% clean accuracy but buys many times more resilience under attack.

## 🔬 Visual experiments / تجارب بصرية

- منحنى FGSM لعدة قيم `epsilon` للموديلين — لاحظ أن منحنى "robust" يهبط بتدرج أبطأ بكثير.
- جدول Clean / FGSM / PGD جنباً إلى جنب.

## 🚀 Quickstart

```bash
pip install -q torch torchvision matplotlib
python defenses_robustness.py
```
> ⏱️ يدرب موديلين (~3 epochs لكل منهما) — أسرع بكثير على T4، ويعمل على CPU.

## ✍️ YOUR TURN missions / تحديات

1. درّب ضد **PGD** بدل FGSM — هل تزداد المتانة؟
2. كبّر `eps` التدريب ولاحظ ثمن الدقة النظيفة.
3. جرّب دفاعاً بسيطاً: تنعيم الصورة قبل التنبؤ — هل يكفي وحده؟

## ➡️ Next steps / بعد السلسلة

Carlini-Wagner & DeepFool · TRADES · certified robustness · CIFAR-10 · قياس كلفة التدريب العدائي.
هذا المشروع هو التطبيق العملي لمقرر **ECE 653 — Machine Learning Security & Privacy**. 🎓

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
