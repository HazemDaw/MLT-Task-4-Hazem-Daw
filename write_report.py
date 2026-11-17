"""Write the README and concise report from actual measurements."""
import json
import hashlib
from pathlib import Path
import pandas as pd
from train import ROOT


def main():
    results = pd.read_csv(ROOT/'results/comparison.csv')
    selection = json.loads((ROOT/'results/selection.json').read_text())
    names = {'cnn':'Three-layer CNN','vgg16':'VGG16 transfer','resnet18':'ResNet18 transfer','scratch':'Residual CNN (contest)'}
    rows = []
    for _,r in results.iterrows():
        rows.append(f"| {names[r.model]} | {'ImageNet' if r.pretrained else 'None'} | {int(r.best_epoch)} | {r.accuracy:.2%} | {r.test_loss:.4f} | {r.ci95_low:.2%} - {r.ci95_high:.2%} |")
    table='| Model | Pretrained weights | Selected epoch | Test accuracy | Test BCE | 95% Wilson interval |\n|---|---|---:|---:|---:|---|\n'+'\n'.join(rows)
    best=selection['best_inference_model']
    scratch=results[results.model=='scratch'].iloc[0]
    readme=f'''# MLT Assignment 4: CNNs and Transfer Learning

**Hazem Daw | Group J4132**  
Machine Learning Technologies, ITMO University, 2026

[Open the executed notebook](CNN_and_Transfer_Learning_Pytorch.ipynb) · [Read the report](REPORT.md)

Cats-versus-dogs classification with the required three-layer CNN, VGG16 and ResNet18 transfer learning, plus a separate residual CNN trained entirely from scratch for the optional contest. All reported results were measured on an NVIDIA RTX 3060; no scores or learning curves are simulated.

## Results

{table}

The inference model is **{names[best]}**, selected by validation BCE before test evaluation. The contest entry is **Residual CNN**, with **{int(scratch.correct)}/2,500 correct ({scratch.accuracy:.2%})**. It uses no pretrained weights. The 5A award is a class competition: this repository does not claim a winning rank.

The reported contest result uses {'horizontal-flip logit averaging' if scratch.tta else 'ordinary single-view inference'}, selected on validation data. To reproduce contest inference from its checkpoint, use `python inference.py IMAGE --checkpoint checkpoints/scratch.pt{' --tta' if scratch.tta else ''}`.

![Model comparison](figures/model_comparison.png)

## Review the saved submission

The notebook contains saved outputs and plots, so GitHub can display the results without running training. To rerun the analysis and custom-image inference locally:

```bash
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements.txt
python download_checkpoints.py
python -m ipykernel install --user --name mlt-task4 --display-name "MLT Task 4"
```

Open `CNN_and_Transfer_Learning_Pytorch.ipynb`, select the environment, and run all cells. Keep `RUN_TRAINING=False` to review the measured run. Saved-results mode needs the repository files and downloaded checkpoints, but does not require the course dataset. The notebook is not a disconnected single-file notebook.

Alternatively, `python run_notebook.py` executes it and exports an HTML copy. This runner includes a PyTorch-before-kernel loading workaround for Windows DLL conflicts. `python validate_results.py` verifies the saved scores, checkpoint hashes and executed notebook.

For NVIDIA CUDA 12.8, install the matching build before the requirements:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

## Reproduce training

```bash
python prepare_data.py
python verify.py
python run_all.py
python build_artifacts.py
python write_report.py
```

`prepare_data.py` downloads the original course archive (about 1.1 GB), checks image decoding and hashes, and preserves the supplied split. `run_all.py` trains all four models, locks validation choices, then evaluates the test set. Training can also be repeated by setting `RUN_TRAINING=True` in the notebook. Internet is required for the course archive and initial torchvision weights. A CUDA GPU is recommended; CPU training is supported but slower.

The recorded experiment budgets are:

```bash
python train.py --model cnn --epochs 15 --batch-size 128
python train.py --model vgg16 --epochs 15 --batch-size 64
python train.py --model resnet18 --epochs 15 --batch-size 64
python train.py --model scratch --epochs 60 --size 128 --batch-size 128
python evaluate.py
```

## Custom-image inference

```bash
python inference.py custom_images/dog.jpg --checkpoint checkpoints/{best}.pt
```

The function converts images to RGB, handles EXIF orientation, applies the checkpoint's preprocessing, and returns a class and both probabilities. Confidence is a model output, not a calibration guarantee.

## Data and reproducibility

- Original counts: 20,000 train, 2,490 validation, 2,500 test; both test classes contain 1,250 images.
- Five duplicate training copies are excluded, leaving **19,995** training images. Validation/test images are retained. Three groups cross split boundaries; two are within training.
- Class map: `cats=0`, `dogs=1`. Logits are trained with BCEWithLogitsLoss; the threshold is fixed at 0.5.
- Seed 42, deterministic cuDNN settings, seeded data-loader workers. Exact results may vary across hardware/library versions.
- Minimum validation BCE selects checkpoints. The scratch model's horizontal-flip test-time augmentation is selected using validation data alone.
- The two ImageNet models are exclusively Part 2; the Part 3 model is initialized randomly and never uses their features or weights.
- Checkpoint files are distributed in the GitHub release, with SHA-256 verification via `download_checkpoints.py`. Dataset and feature caches are not committed.

## Files

| File/folder | Purpose |
|---|---|
| `CNN_and_Transfer_Learning_Pytorch.ipynb` | Completed exercise, explanations, implementations and saved outputs |
| `models.py`, `train.py` | Model definitions and reusable training engine |
| `prepare_data.py` | Safe extraction, image checks and duplicate audit |
| `evaluate.py`, `inference.py` | Locked evaluation and custom-image prediction |
| `results/` | Configurations, epoch histories, data hashes, test predictions and metrics |
| `figures/` | Learning curves, confusion matrices, comparison and error analysis |
| `verify.py` | Focused numerical and model-behavior checks |
| `custom_images/` | External inference example and provenance |

Sources: supplied Task 4 PDF, lecture 7 and course PyTorch notebook; [torchvision VGG16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html), [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), and the [course dataset](https://drive.google.com/drive/folders/1KpT0oBIb3aDaBtjf7Hs4WAUPoFEpbsYr).
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    report=f'''# Assignment 4: experimental report

**Hazem Daw | J4132**

## Objective and protocol

The task is binary classification of cat and dog photographs. The provided split is retained, with five duplicate training copies removed using exact decoded-pixel hashes before fitting. There are 19,995 training images, 2,490 validation images and 2,500 test images. No image failed decoding. The archive contains ten fewer validation cats than the 25,000-image total described by the example notebook.

Model weights use only training data. Validation BCE selects the checkpoint and whether the contest model uses horizontal-flip logit averaging. Architecture, training budgets and inference choices are fixed before final test evaluation. The supplied test set contains two identical dog photographs; confidence intervals therefore approximate the independent-sample setting.

## Architectures and optimization

**Baseline:** the specified 16/32/64-channel CNN, three 3x3 convolutions with ReLU and max pooling, then 64 hidden units, dropout 0.5 and one logit. Input is 150x150 with pixels in [0,1]. It trains for 15 epochs at learning rate 0.001.

**VGG16:** frozen ImageNet V1 convolutional features, adaptive 7x7 pooling and a new 25,088 -> 256 -> 1 head with dropout 0.5. Only the head trains, for 15 epochs at 0.0001.

**ResNet18:** frozen ImageNet V1 features and a new 512 -> 256 -> 1 head. BatchNorm running statistics remain frozen. The head trains for 15 epochs at 0.001. Both transfer models use torchvision's weight-specific 224-pixel preprocessing. Deterministic float16 feature caches save repeated backbone passes; training promotes features to float32.

**Contest:** a residual CNN with channels 32/64/128/256, global average pooling and dropout 0.2. Every parameter starts from random initialization. It trains for 60 epochs at 128x128 with random resized crops, flips, mild color jitter, label smoothing 0.05, AdamW (initial learning rate 0.002, weight decay 0.0001), and cosine decay. Its training loss uses smoothed labels; validation/test losses use hard labels. No pretrained features, weights, distillation or external training images are used.

## Measured results

{table}

![Comparison](figures/model_comparison.png)

The validation-selected inference model is **{names[best]}**. The contest-eligible result is **{scratch.accuracy:.2%}**, or **{int(scratch.correct)} correct predictions out of 2,500**. These are separate categories: pretrained models cannot qualify for Part 3. A constant-class predictor scores 50% on this balanced test set.

## Interpretation and limits

The baseline's training accuracy improves beyond the epoch with minimum validation loss, illustrating why the final epoch is not automatically the best checkpoint. Cross-entropy also penalizes confidently wrong predictions, so its optimum need not coincide with maximum accuracy. Frozen pretrained features exploit a much larger upstream training dataset; this comparison does not isolate architecture as the only cause of differences.

The notebook includes confusion matrices, both accuracy and loss curves for every model, and the selected model's most confident errors. Several shown mistakes contain both species, making a single class label ambiguous; others have small subjects or poor lighting. The original test labels were retained. Error inspection was performed after fitting and was not fed back into model tuning. A separately sourced dog photo demonstrates inference. It is not a statistical evaluation, and overlap between that photograph and ImageNet cannot be ruled out.

There is one training seed per architecture. Wilson intervals describe test-sample uncertainty only, not training-seed variation. Class ranking and other students' scores are unavailable, so an automatic 5A cannot be established from these experiments alone. Higher-resolution training, multiple scratch seeds and validation-selected ensembling are potential follow-ups, not measured claims in this submission.

## Reproducibility evidence

`results/selection.json` records decisions and checkpoint hashes before test evaluation. Per-image probabilities are saved in each model's `test_predictions.csv`. `data_manifest.csv` records every image's split, class, dimensions and file/pixel SHA-256. Configurations and all per-epoch measurements are saved. `verify.py` checks logits, gradient flow, frozen BatchNorm, weighted evaluation, and image conversion. See README for exact commands.
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    manifest={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,
                         url=f'https://github.com/HazemDaw/MLT-Task-4-Hazem-Daw/releases/download/v1.0/{p.name}')
              for p in sorted((ROOT/'checkpoints').glob('*.pt'))}
    (ROOT/'results/checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
