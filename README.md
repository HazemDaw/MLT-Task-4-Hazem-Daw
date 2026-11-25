# Assignment 4: CNNs and Transfer Learning

**Hazem Daw | Group J4132**

Machine Learning Technologies, ITMO University, 2026

**Part 3 contest entry: 2,452/2,500 correct (98.08%) on the course test set.** The entry averages the predictions of EfficientNetV2-S and the earlier residual CNN classifier. Both were trained from random initialization on the course images.

- [Assignment notebook](CNN_and_Transfer_Learning_Pytorch.ipynb) — Parts 1 and 2, custom-image inference and the original Part 3 baseline.
- [Contest notebook](EfficientNetV2_S_Contest.ipynb) — EfficientNetV2-S training, validation comparisons and the final test result. To reproduce the evaluation, [download the notebook with its trained weights](https://github.com/HazemDaw/MLT-Task-4-Hazem-Daw/releases/download/v1.1/EfficientNetV2_S_with_Weights.zip).

## Assignment coverage

| Requirement | Notebook section |
|---|---|
| Train a CNN with three convolutional layers | Part 1: Three-layer CNN |
| Test accuracy and loss/accuracy learning curves | Part 1 and Final test evaluation |
| VGG transfer learning and another torchvision model | Part 2: VGG16 and ResNet18 |
| Test accuracies and learning curves for both transfer models | Part 2 and Final test evaluation |
| Custom-image test and reusable inference function | Custom-image inference; `predict_image` |
| Optional classifier without pretrained weights | [Contest notebook](EfficientNetV2_S_Contest.ipynb) |

## Results

All models use the same 2,500-image test split. Checkpoints and inference settings are selected on validation data before test evaluation.

| Model | Pretrained weights | Correct / 2,500 | Test accuracy | Test BCE |
|---|---|---:|---:|---:|
| Three-layer CNN | None | 2,054 | 82.16% | 0.3842 |
| VGG16 transfer | ImageNet | 2,458 | 98.32% | 0.0481 |
| ResNet18 transfer | ImageNet | 2,464 | 98.56% | 0.0433 |
| Residual CNN | None | 2,423 | 96.92% | 0.1064 |
| **Contest entry · EfficientNetV2-S + residual CNN classifier** | **None** | **2,452** | **98.08%** | **0.0731** |

ResNet18 is selected for custom-image inference by validation BCE. For the contest, EfficientNetV2-S trained for 48 epochs at 160 pixels and 16 epochs at 224 pixels. Its best standalone inference setting used the full image and corner crops and classified 2,449 of 2,490 validation images correctly. An equal average of its logits and the higher-resolution residual CNN classifier's logits scored 2,450/2,490 on validation. This setting improved the previous contest test score from 2,449 to 2,452 correct. [Validation selection](results/efficientnet/eval_selection.json) · [Test predictions](results/efficientnet/eval_test.csv)

## Running

```bash
python -m pip install -r requirements.txt
```

Both notebooks display saved outputs on GitHub.

In the **assignment notebook**, Run All repeats data preparation, training and evaluation, including the original 60-epoch optional experiment. It downloads the course data and Part 2 pretrained weights as needed, then saves generated files under `task4_runtime/`.

For the **contest notebook**, extract the [v1.1 bundle](https://github.com/HazemDaw/MLT-Task-4-Hazem-Daw/releases/tag/v1.1) and open the notebook inside it. Keep `reference_weights/` and `efficientnet_runtime/` next to the notebook. Run All uses the saved checkpoints to reproduce the comparison. To repeat the 64 training epochs, set `RUN` to a fresh directory. Set `DATA_ROOT` to the extracted course dataset or let the notebook download it.

For custom-image inference, set `CUSTOM_IMAGE_PATH` to another photograph, or leave it as `None` for the example image. Inference uses the checkpoint selected by validation BCE.

The recorded run used Python 3.11, PyTorch 2.11.0, torchvision 0.26.0 and an NVIDIA RTX 3060. Retraining is supported on CPU but is substantially slower. Windows notebook data loaders use zero workers; worker count and hardware differences can change exact results.

On Windows, use the listed `pyzmq>=27` dependency and restart the kernel after installation.

## Data and validation

The supplied archive contains 20,000 training, 2,490 validation and 2,500 test images. Five duplicate training copies were excluded after a decoded-pixel hash audit, leaving 19,995 training images. Validation and test images were retained. Class mapping is `cats=0`, `dogs=1`.

The reported scores were checked against the saved per-image predictions. The complete cell sequence was tested in a temporary copy using small real-data subsets and one epoch per model. The submission retains the full experiment outputs; the short verification results are kept separate. Details are in [submission checks](results/submission_checks.json).

## Repository files

| Item | Contents |
|---|---|
| `CNN_and_Transfer_Learning_Pytorch.ipynb` | Assignment experiments, custom-image inference and original contest baseline |
| `EfficientNetV2_S_Contest.ipynb` | Final contest entry, training curves and recorded results |
| `requirements.txt` | Python dependencies |
| `results/` | Configurations, histories, per-image predictions, data audit and validation records |

Contest weights are in the [v1.1 release](https://github.com/HazemDaw/MLT-Task-4-Hazem-Daw/releases/tag/v1.1). The [v1.0 release](https://github.com/HazemDaw/MLT-Task-4-Hazem-Daw/releases/tag/v1.0) contains the original assignment checkpoints.

Sources: supplied Task 4 assignment sheet, lecture 7 and course PyTorch notebook; [course dataset](https://drive.google.com/drive/folders/1KpT0oBIb3aDaBtjf7Hs4WAUPoFEpbsYr), [VGG16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html), and [ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
