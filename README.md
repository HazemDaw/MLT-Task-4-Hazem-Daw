# Task 4 — One Notebook

**Hazem Daw · Group J4132 · ITMO MLT 2026**

## افتح هذا الملف فقط

### [CNN_and_Transfer_Learning_Pytorch.ipynb](CNN_and_Transfer_Learning_Pytorch.ipynb)

هذا هو التسليم الكامل: **الكود + الشرح + الصور + الرسومات + النتائج**، داخل Notebook واحد.
لا تحتاج ملفات Python الموجودة في المجلد المساعد، ولا تحتاج تنزيل البيانات لمشاهدة النتائج أو إعادة عرضها.

| داخل الـNotebook | الشغل | الدقة على الاختبار |
|---|---|---:|
| **Part 1 — Required** | شبكة CNN من 3 طبقات، حسب المطلوب | 82.16% |
| **Part 2 — Required** | VGG16 وResNet18 بالتعلم بالنقل | 98.32% / **98.56%** |
| **Part 3 — 5A Contest** | نموذجنا المدرّب من الصفر، بدون أوزان جاهزة | **96.32%** |

الـ5A التلقائية مرتبطة بأعلى نتيجة مؤهلة في المسابقة، وليست مضمونة بتسليم الواجب. نتيجة المسابقة الخاصة بنا هي 96.32%؛ نموذج ResNet18 الجاهز لا يدخل هذه المسابقة.

## One-file submission

Download or submit the notebook above. It contains the complete implementation and embedded images/results.
It was executed successfully in a folder containing **only that notebook**: 19 code cells, three embedded photo panels and six generated plots.
The file is approximately 1.65 MB, reduced from the previous 6.14 MB version.

- **Read:** open the saved notebook on GitHub, Jupyter, VS Code or Colab.
- **Run All to review:** leave `RUN_TRAINING=False` and `RUN_CUSTOM_INFERENCE=False`. No project files, dataset download or checkpoint download are needed.
- **Retrain:** set `RUN_TRAINING=True`. All training code is inside the notebook. The course dataset and Part 2 pretrained weights download automatically into `task4_runtime/`.
- **Predict another image:** enable `RUN_CUSTOM_INFERENCE` and set `CUSTOM_IMAGE_PATH`. The selected trained checkpoint downloads automatically with SHA-256 verification.

Standard Python packages are still required to run code: PyTorch/torchvision, NumPy, pandas, Matplotlib, Pillow and scikit-learn; gdown is needed only for data download. Setup instructions are inside the notebook.

[Trained checkpoint release](https://github.com/HazemDaw/MLT-Task-4-Hazem-Daw/releases/tag/v1.0)

The `supporting_files/` folder preserves the original modular code, detailed logs, per-image predictions and audit records for reference. **It is optional and is not required by the notebook.** The measured results and released weights are unchanged.
