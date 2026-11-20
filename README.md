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

Open the notebook to read the explanations and saved outputs. Viewing it does not require training or downloading any additional files.

**Run All repeats data preparation, training and evaluation**, including the 60-epoch optional experiment. The notebook downloads the course data and Part 2 pretrained weights as needed, then saves generated files under `task4_runtime/`. Run the cells in order and allow time for training.

For custom-image inference, set `CUSTOM_IMAGE_PATH` to another photograph, or leave it as `None` for the example image. Inference uses the checkpoint selected by validation BCE.

The recorded run used Python 3.11, PyTorch 2.11.0, torchvision 0.26.0 and an NVIDIA RTX 3060. Retraining is supported on CPU but is substantially slower. Windows notebook data loaders use zero workers; worker count and hardware differences can change exact results.

On Windows, use the listed `pyzmq>=27` dependency and restart the kernel after installation.

## Data and validation

The supplied archive contains 20,000 training, 2,490 validation and 2,500 test images. Five duplicate training copies were excluded after a decoded-pixel hash audit, leaving 19,995 training images. Validation and test images were retained. Class mapping is `cats=0`, `dogs=1`.

The reported scores were checked against the saved per-image predictions. The complete cell sequence was tested in a temporary copy using small real-data subsets and one epoch per model. The submission retains the full experiment outputs; the short verification results are kept separate. Details are in [submission checks](results/submission_checks.json).

## Repository files

| Item | Contents |
|---|---|
| `CNN_and_Transfer_Learning_Pytorch.ipynb` | Complete submission, including embedded images and outputs |
| `requirements.txt` | Python dependencies |
| `results/` | Configurations, histories, per-image predictions, data audit and validation records |

The `results/` files document the completed experiments. The notebook is self-contained: it displays saved outputs when opened and generates new results when run. Trained weights are also available in the [checkpoint release](https://github.com/HazemDaw/MLT-Task-4-Hazem-Daw/releases/tag/v1.0).

Sources: supplied Task 4 assignment sheet, lecture 7 and course PyTorch notebook; [course dataset](https://drive.google.com/drive/folders/1KpT0oBIb3aDaBtjf7Hs4WAUPoFEpbsYr), [VGG16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html), and [ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
