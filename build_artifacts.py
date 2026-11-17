"""Build figures and an executed, review-friendly notebook from measured runs."""
import inspect
import json
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nbformat
from PIL import Image
from train import ROOT
from models import CNNModel, TransferModel, ResidualBlock, ScratchResNet

COLORS = ['#176B87', '#E17C37', '#5D4B91', '#26806A']
NAMES = {'cnn':'Three-layer CNN', 'vgg16':'VGG16 transfer', 'resnet18':'ResNet18 transfer', 'scratch':'Residual CNN from scratch'}


def build_figures():
    out = ROOT / 'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'axes.titleweight':'bold', 'figure.dpi':130,
                         'savefig.dpi':170, 'axes.grid':True, 'grid.alpha':0.16})
    comparison = pd.read_csv(ROOT / 'results/comparison.csv')
    for _, row in comparison.iterrows():
        name = row['model']
        h = pd.read_csv(ROOT / 'results' / name / 'history.csv')
        fig, axes = plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
        for ax, metric, label in zip(axes,['loss','accuracy'],['Binary cross-entropy','Accuracy']):
            ax.plot(h.epoch,h['train_'+metric],label='Training',color=COLORS[0],lw=2)
            ax.plot(h.epoch,h['val_'+metric],label='Validation',color=COLORS[1],lw=2)
            ax.axvline(row.best_epoch,color='#888888',ls=':',label=f'Selected epoch {int(row.best_epoch)}')
            ax.set(xlabel='Epoch',ylabel=label,title=label)
            ax.legend(fontsize=9)
        fig.suptitle(NAMES.get(name,name))
        fig.savefig(out / f'{name}_learning_curves.png')
        plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for ax, (_, row) in zip(axes.flat,comparison.iterrows()):
        metrics = json.loads((ROOT / 'results' / row.model / 'test_metrics.json').read_text())
        cm = np.array(metrics['classification_report']['confusion_matrix'])
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=1250)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=19,
                        color='white' if cm[i,j]>600 else '#123047')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Cat','Dog'],yticklabels=['Cat','Dog'],
               xlabel='Predicted class',ylabel='True class',title=NAMES.get(row.model,row.model))
        ax.grid(False)
    fig.savefig(out / 'confusion_matrices.png')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,4.7),layout='constrained')
    y = np.arange(len(comparison))
    x = comparison.accuracy.to_numpy()*100
    err = np.array([x-comparison.ci95_low.to_numpy()*100,comparison.ci95_high.to_numpy()*100-x])
    ax.barh(y,x,color=COLORS[:len(y)],height=0.55,alpha=0.9)
    ax.errorbar(x,y,xerr=err,fmt='none',ecolor='#202938',capsize=4)
    for i,score in enumerate(x): ax.text(1,i,f'{score:.2f}%',va='center',color='white',weight='bold')
    ax.set(yticks=y,yticklabels=[NAMES.get(n,n) for n in comparison.model],xlim=(0,100),
           xlabel='Test accuracy (%) with 95% Wilson interval',title='Final evaluation on 2,500 course test images')
    ax.invert_yaxis()
    fig.savefig(out / 'model_comparison.png')
    plt.close(fig)
    selected=json.loads((ROOT/'results/selection.json').read_text())['best_inference_model']
    preds=pd.read_csv(ROOT/'results'/selected/'test_predictions.csv')
    preds['confidence']=np.maximum(preds.probability_dog,1-preds.probability_dog)
    errors=preds[preds.target!=preds.prediction].sort_values('confidence',ascending=False).head(8)
    fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    for ax,(_,r) in zip(axes.flat,errors.iterrows()):
        with Image.open(ROOT/'data/Cats_and_Dogs'/r.path) as im: ax.imshow(im.convert('RGB'))
        ax.set_title(f"True: {['cat','dog'][int(r.target)]} | predicted: {['cat','dog'][int(r.prediction)]}\nConfidence: {r.confidence:.1%}",fontsize=10)
        ax.axis('off')
    for ax in axes.flat[len(errors):]: ax.axis('off')
    fig.suptitle(f'Confident errors: {NAMES.get(selected,selected)}')
    fig.savefig(out/'error_analysis.png')
    plt.close(fig)
    train_paths=list((ROOT/'data/Cats_and_Dogs/train/cats').glob('*.jpg'))[:4]+list((ROOT/'data/Cats_and_Dogs/train/dogs').glob('*.jpg'))[:4]
    fig,axes=plt.subplots(2,4,figsize=(10,5),layout='constrained')
    for ax,path in zip(axes.flat,train_paths):
        with Image.open(path) as im: ax.imshow(im.convert('RGB'))
        ax.set_title(path.parent.name)
        ax.axis('off')
    fig.suptitle('Examples from the training split')
    fig.savefig(out/'training_examples.png')
    plt.close(fig)


def build_notebook():
    c=[]
    md=lambda s:c.append(nbformat.v4.new_markdown_cell(textwrap.dedent('    '+s).strip()))
    code=lambda s:c.append(nbformat.v4.new_code_cell(textwrap.dedent('    '+s).strip()))
    raw_code=lambda s:c.append(nbformat.v4.new_code_cell(s.strip()))
    md('''# Assignment 4: Cats and Dogs with Convolutional Neural Networks
    **Hazem Daw | Group J4132**  
    Machine Learning Technologies, ITMO University, 2026

    This completes the supplied CNN and transfer-learning exercise, including the optional
    contest. All numbers below come from actual GPU runs. The training engine lives in
    `train.py`; the model definitions live in `models.py`. This notebook explains the method,
    displays the implementations, and reproduces the analysis from saved outputs.

    **Contest eligibility:** only the residual CNN marked **from scratch** is an entry for
    Part 3. VGG16 and ResNet18 use ImageNet weights for Part 2 and are ineligible for the contest.
    The automatic 5A award depends on the class ranking, which is not available here.''')
    code('''from pathlib import Path
    import json, inspect
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Image as DisplayImage
    from torch import nn
    from torchvision import models
    from models import CNNModel, TransferModel, ScratchResNet
    from train import train_run, preprocessing, remove_training_duplicates
    from inference import load_classifier, predict_image
    ROOT = Path.cwd()
    RUN_TRAINING = False  # True repeats the measured runs; False reviews saved results.
    if RUN_TRAINING:
        import subprocess, sys
        subprocess.run([sys.executable, 'run_all.py'], check=True)
    assert (ROOT / 'results/comparison.csv').exists(), 'Run from the repository folder.'
    print('PyTorch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    comparison = pd.read_csv(ROOT / 'results/comparison.csv')''')
    md('''## 1. Data, checks and evaluation protocol
    The original course archive has 20,000 training, 2,490 validation and 2,500 test images.
    The balanced test set gives a constant-class baseline of 50% accuracy.
    It contains 24,990 images, not the 25,000 mentioned in the example notebook: validation
    has 1,240 cats and 1,250 dogs. No unreadable image was silently skipped.

    SHA-256 hashes of both files and decoded RGB pixels are recorded in `results/data_manifest.csv`.
    Three duplicate groups cross split boundaries. Their training copies are excluded;
    two redundant copies within training are also excluded, leaving **19,995 training images**.
    The supplied validation and test splits remain intact. Two test dog images are identical,
    so the binomial confidence intervals are descriptive approximations, not exact independent-sample guarantees.

    ImageFolder maps **cats = 0, dogs = 1**. All networks return logits. `BCEWithLogitsLoss`
    applies the stable sigmoid/BCE combination; adding a sigmoid in the model would apply it twice.
    A logit of zero corresponds to the fixed probability threshold 0.5.

    Only training images update weights. Minimum validation BCE selects the epoch. The test
    set is evaluated after architecture, training budget, checkpoint and flip-augmentation
    choices have been fixed. Audit hashing does not use image labels for model selection.''')
    code('''audit = json.loads((ROOT / 'results/data_audit.json').read_text())
    display(pd.Series(audit['counts'], name='Images').to_frame())
    print('Cross-split duplicate groups:', len(audit['cross_split_duplicates']))
    display(DisplayImage(filename=str(ROOT/'figures/training_examples.png')))
    print(inspect.getsource(remove_training_duplicates))''')
    md('''## 2. Part 1: the required three-layer CNN
    The baseline follows the supplied architecture: 16, 32 and 64 convolutional channels,
    3x3 kernels, ReLU and 2x2 max pooling after each convolution. At 150x150 input the spatial
    sizes are 150 -> 75 -> 37 -> 18. Flattening feeds 64 hidden units, dropout 0.5 and one logit.
    Pixels are scaled to [0,1]. AdamW with zero weight decay is equivalent here to Adam;
    the initial learning rate is 0.001.''')
    raw_code(inspect.getsource(CNNModel))
    code('''model = CNNModel()
    print('Output shape:', tuple(model(torch.zeros(2,3,150,150)).shape))
    print('Parameters:', sum(p.numel() for p in model.parameters()))
    del model''')
    code('''config = json.loads((ROOT/'results/cnn/config.json').read_text())
    display(pd.read_csv(ROOT/'results/cnn/history.csv'))
    display(DisplayImage(filename=str(ROOT/'figures/cnn_learning_curves.png')))''')
    md('''### Training and validation implementation
    The engine switches between train/eval modes, clears gradients for each batch, uses
    mixed precision on CUDA, clips gradient norms, and computes metrics over all examples.
    Validation uses `inference_mode`; the last partial batch receives its correct weight.
    Dropout makes training and validation metrics differ even without overfitting.
    The saved histories are measurements, not illustrative or simulated curves.''')
    code('''print(inspect.getsource(train_run))''')
    md('''## 3. Part 2: transfer learning with VGG16 and ResNet18
    Both feature extractors use torchvision's explicit `IMAGENET1K_V1` weights and their
    official inference transforms: resize the shorter edge to 256, center-crop to 224,
    scale pixels and normalize using ImageNet mean/std. The VGG head is the requested
    512x7x7 -> 256 -> 1 architecture with ReLU and dropout 0.5. The ResNet head is
    512 -> 256 -> 1 with the same activation/dropout.

    The frozen backbones stay in eval mode, including ResNet BatchNorm. Deterministic
    features are computed once, cached as float16, and promoted to float32 for head training.
    This is mathematically the same frozen-feature workflow apart from float16 rounding;
    it avoids repeatedly computing an unchanged backbone. Transfer training uses no random
    augmentation. Only the new heads learn. VGG uses learning rate 0.0001 and ResNet 0.001.''')
    raw_code(inspect.getsource(TransferModel))
    code('''for name in ['vgg16','resnet18']:
        config = json.loads((ROOT/'results'/name/'config.json').read_text())
        display(pd.Series(config).to_frame(name))
        display(DisplayImage(filename=str(ROOT/'figures'/f'{name}_learning_curves.png')))''')
    md('''## 4. Part 3: contest model trained from scratch
    The compact residual CNN below has no pretrained weights and uses no teacher,
    pretrained feature cache, or pretrained ensemble member. Kaiming initialization and
    residual connections support optimization. Training uses random resized crops,
    horizontal flips, mild color jitter, dropout, AdamW weight decay 0.0001 and a cosine
    learning-rate schedule. Normalization uses fixed mean/std 0.5, not fitted test statistics.

    Label smoothing maps binary targets to 0.025 and 0.975 during training; validation and
    test BCE always use hard labels. Thus the scratch model's training loss and validation
    loss have different targets and should not be interpreted as a direct generalization gap.
    Validation compares ordinary inference with averaging the original and horizontally
    flipped logits; the lower validation BCE decides whether flip averaging is used.''')
    raw_code(inspect.getsource(ResidualBlock)+'\n\n'+inspect.getsource(ScratchResNet))
    code('''config = json.loads((ROOT/'results/scratch/config.json').read_text())
    display(pd.Series(config).to_frame('Contest configuration'))
    display(DisplayImage(filename=str(ROOT/'figures/scratch_learning_curves.png')))''')
    md('''## 5. Locked selection and final test evaluation
    Test accuracy answers the required question; BCE measures probability quality,
    ROC-AUC measures ranking, and F1 measures the dog-class precision/recall tradeoff.
    Wilson intervals show uncertainty for each accuracy estimate and are not a paired
    significance test. This is one seeded training run per architecture; it does not estimate
    training-seed variability. ImageNet pretraining has a different data budget, so its
    advantage must not be attributed solely to architecture.''')
    code('''selection = json.loads((ROOT/'results/selection.json').read_text())
    comparison = pd.read_csv(ROOT/'results/comparison.csv')
    decisions = pd.DataFrame(selection['models']).T
    display(decisions[['best_epoch','validation_accuracy','validation_loss','tta']])
    display(comparison[['model','pretrained','best_epoch','accuracy','test_loss','roc_auc','tta']]
            .style.format({'accuracy':'{:.2%}', 'test_loss':'{:.4f}', 'roc_auc':'{:.4f}'}))
    print('Best inference model, selected using validation BCE:', selection['best_inference_model'])
    display(DisplayImage(filename=str(ROOT/'figures/model_comparison.png')))
    display(DisplayImage(filename=str(ROOT/'figures/confusion_matrices.png')))''')
    md('''### Error analysis
    The following are the most confident mistakes of the validation-selected model.
    They help diagnose remaining limitations after evaluation; they were not used to
    revise the fitted models. High sigmoid confidence is not proof of correctness or
    calibrated probability. Cropping can remove context, and unusual poses/backgrounds
    can differ from the training distribution. Several high-confidence errors shown here
    contain both a cat and a dog, making the single dataset label ambiguous. Other examples
    involve small subjects or poor lighting. The original labels were kept for scoring.''')
    code('''display(DisplayImage(filename=str(ROOT/'figures/error_analysis.png')))''')
    md('''## 6. Custom image and reusable inference
    The example dog photograph comes from the PyTorch Hub repository, outside the course
    archive. It is an inference demonstration, not another statistical test set; possible
    overlap with ImageNet cannot be ruled out. Its source and hash are recorded in
    `custom_images/README.md`. The function handles EXIF orientation and RGB conversion,
    uses the checkpoint's class map, and returns both class probabilities.''')
    code('''print(inspect.getsource(predict_image))
    best = selection['best_inference_model']
    model, config, device = load_classifier(ROOT/'checkpoints'/f'{best}.pt')
    result = predict_image(ROOT/'custom_images/dog.jpg', model, config, device,
                           tta=selection['models'][best]['tta'])
    display(DisplayImage(filename=str(ROOT/'custom_images/dog.jpg'), width=400))
    print(json.dumps(result, indent=2))
    assert abs(sum(result['probabilities'].values()) - 1) < 1e-6
    del model
    torch.cuda.empty_cache()''')
    md('''## 7. Conclusions and reproducibility
    The table and curves above separate the required baseline, transfer learning, and
    the eligible contest model. Transfer learning can reuse features learned from a much
    larger dataset; the scratch model must learn its own representation from 19,995 images.
    The selected epoch may precede the final epoch because validation probability quality
    can deteriorate even while training accuracy rises.

    To reproduce: install `requirements.txt`, run `prepare_data.py`, and either set
    `RUN_TRAINING=True` or run the training commands in README. Checkpoints are supplied
    through the repository release and can also be regenerated. Saved metrics, histories,
    predictions, configurations, data hashes and checkpoint hashes make the run auditable.
    Fixed seeds improve repeatability; CUDA/kernel/library differences can still cause
    numerical changes. The contest score is a measured result, not a guarantee of winning.

    **References:** the supplied Task 4 PDF, Lecture 7 and PyTorch course notebook;
    [torchvision models](https://docs.pytorch.org/vision/stable/models.html),
    [VGG16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html),
    [ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).''')
    nb=nbformat.v4.new_notebook(cells=c,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
                                                'language_info':{'name':'python','version':'3.11'}})
    nbformat.write(nb,ROOT/'CNN_and_Transfer_Learning_Pytorch.ipynb')


if __name__=='__main__':
    build_figures()
    build_notebook()
