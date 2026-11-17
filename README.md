# Assignment 4: CNNs and Transfer Learning

**Hazem Daw | Group J4132**

Machine Learning Technologies, ITMO University, 2026

Cats-versus-dogs classification using a three-layer CNN, VGG16 and ResNet18 transfer learning, and a residual CNN trained from scratch.

**[Open the notebook](CNN_and_Transfer_Learning_Pytorch.ipynb)** for the complete implementation, explanations, images and saved outputs. All project code is contained in this file.

## Assignment coverage

| Requirement | Notebook section |
|---|---|
| Train a CNN with three convolutional layers | Part 1: Three-layer CNN |
| Test accuracy and loss/accuracy learning curves | Part 1 and Final test evaluation |
| VGG transfer learning and another torchvision model | Part 2: VGG16 and ResNet18 |
| Test accuracies and learning curves for both transfer models | Part 2 and Final test evaluation |
| Custom-image test and reusable inference function | Custom-image inference; `predict_image` |
| Optional classifier without pretrained weights | Part 3: Residual CNN trained from scratch |

## Results

All models use the same 2,500-image test split. Checkpoints and inference settings are selected on validation data before test evaluation.

| Model | Pretrained weights | Correct / 2,500 | Test accuracy | Test BCE |
|---|---|---:|---:|---:|
| Three-layer CNN | None | 2,054 | 82.16% | 0.3842 |
| VGG16 transfer | ImageNet | 2,458 | 98.32% | 0.0481 |
| ResNet18 transfer | ImageNet | 2,464 | 98.56% | 0.0433 |
| Residual CNN | None | 2,408 | 96.32% | 0.1137 |

ResNet18 is selected for custom-image inference by validation BCE. The residual CNN is the optional contest entry; its result uses validation-selected horizontal-flip logit averaging. The pretrained models are excluded from the contest.

## Running

```bash
python -m pip install -r requirements.txt
```

Open the notebook in Jupyter or Colab and run the cells in order.

- With `RUN_TRAINING=False` and `RUN_CUSTOM_INFERENCE=False`, tables and plots are reproduced from stored measurements. No accompanying project files, dataset download or checkpoint download are required.
- Set `RUN_TRAINING=True` to repeat training and evaluation. The notebook downloads the course data and the Part 2 pretrained weights as needed. Files are saved under `task4_runtime/`.
- Set `RUN_CUSTOM_INFERENCE=True` to predict another image. Set `CUSTOM_IMAGE_PATH` to its path, or leave it as `None` for the included example. The selected checkpoint is downloaded and its SHA-256 is checked.

The recorded run used Python 3.11, PyTorch 2.11.0, torchvision 0.26.0 and an NVIDIA RTX 3060. Retraining is supported on CPU but is substantially slower. Windows notebook data loaders use zero workers; worker count and hardware differences can change exact results.

On Windows, use the listed `pyzmq>=27` dependency and restart the kernel after installation. An older pyzmq installation on the verification machine loaded an incompatible runtime DLL before PyTorch. The notebook passed all cells with the standard Jupyter kernel after updating pyzmq to 27.3.0.

## Data and validation

The supplied archive contains 20,000 training, 2,490 validation and 2,500 test images. Five duplicate training copies were excluded after a decoded-pixel hash audit, leaving 19,995 training images. Validation and test images were retained. Class mapping is `cats=0`, `dogs=1`.

The submitted notebook was executed independently, and its scores were checked against the saved per-image predictions. Additional checks covered tensor shapes, gradients, frozen BatchNorm statistics, class probabilities and all four training paths on small real-data subsets. These short training checks are separate from the full recorded experiments. Details are in [submission checks](results/submission_checks.json).

## Repository files

| Item | Contents |
|---|---|
| `CNN_and_Transfer_Learning_Pytorch.ipynb` | Complete submission, including embedded images and outputs |
| `requirements.txt` | Python dependencies |
| `results/` | Configurations, histories, per-image predictions, data audit and validation records |

The `results/` files provide additional evidence; the notebook does not need them in saved-results mode. Trained weights are available in the [checkpoint release](https://github.com/HazemDaw/MLT-Task-4-Hazem-Daw/releases/tag/v1.0).

Sources: supplied Task 4 assignment sheet, lecture 7 and course PyTorch notebook; [course dataset](https://drive.google.com/drive/folders/1KpT0oBIb3aDaBtjf7Hs4WAUPoFEpbsYr), [VGG16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html), and [ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
