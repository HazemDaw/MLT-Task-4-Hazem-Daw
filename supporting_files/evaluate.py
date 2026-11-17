"""Lock validation selections before evaluating the held-out course test set."""
import json
import math
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from torchvision import datasets
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from inference import load_classifier, predict_image
from train import ROOT, preprocessing, loader, evaluate, seed_everything


def wilson(correct, total):
    z = 1.959963984540054
    p = correct / total
    d = 1 + z*z / total
    center = (p + z*z / (2*total)) / d
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / d
    return center-half, center+half


def main():
    seed_everything(42)
    paths = sorted((ROOT / 'checkpoints').glob('*.pt'))
    selections = {}
    for path in paths:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        selections[path.stem] = dict(name=saved['config']['name'],
                                     best_epoch=saved['best_epoch'],
                                     validation_loss=saved['validation_loss'],
                                     validation_accuracy=saved['validation_accuracy'],
                                     checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    # Test-time augmentation for the contest is selected solely on validation BCE.
    for key, selected in selections.items():
        selected['tta'] = False
        if selected['name'] == 'scratch':
            model, config, device = load_classifier(ROOT / 'checkpoints' / f'{key}.pt')
            val_ds = datasets.ImageFolder(ROOT / 'data/Cats_and_Dogs/val', preprocessing('scratch', size=config['image_size']))
            batches = loader(val_ds, workers=4)
            plain = evaluate(model, batches, device)
            tta = evaluate(model, batches, device, tta=True)
            selected['validation_plain_loss'] = plain['loss']
            selected['validation_tta_loss'] = tta['loss']
            selected['tta'] = tta['loss'] < plain['loss']
            selected['validation_loss'] = min(plain['loss'], tta['loss'])
            selected['validation_accuracy'] = tta['accuracy'] if selected['tta'] else plain['accuracy']
            del model, batches
            torch.cuda.empty_cache()
    best = min(selections, key=lambda k: selections[k]['validation_loss'])
    locked = dict(best_inference_model=best, selection_rule='lowest validation BCE before test access', models=selections)
    (ROOT / 'results/selection.json').write_text(json.dumps(locked, indent=2))
    rows = []
    for key, selected in selections.items():
        model, config, device = load_classifier(ROOT / 'checkpoints' / f'{key}.pt')
        test_ds = datasets.ImageFolder(ROOT / 'data/Cats_and_Dogs/test', preprocessing(config['name'], size=config['image_size']))
        assert test_ds.class_to_idx == config['class_to_idx']
        result = evaluate(model, loader(test_ds, workers=4), device, tta=selected['tta'])
        y, p = result['y'].astype(int), result['probabilities']
        pred = (p >= 0.5).astype(int)
        correct = int((pred == y).sum())
        low, high = wilson(correct, len(y))
        row = dict(model=key, pretrained=config['pretrained'], best_epoch=selected['best_epoch'],
                   validation_loss=selected['validation_loss'], test_loss=result['loss'],
                   accuracy=accuracy_score(y,pred), balanced_accuracy=balanced_accuracy_score(y,pred),
                   f1_dog=f1_score(y,pred), roc_auc=roc_auc_score(y,p), correct=correct,
                   test_images=len(y), ci95_low=low, ci95_high=high, tta=selected['tta'])
        rows.append(row)
        dest = ROOT / 'results' / key
        pd.DataFrame(dict(path=[Path(x[0]).relative_to(ROOT / 'data/Cats_and_Dogs').as_posix() for x in test_ds.samples],
                          target=y, probability_dog=p, prediction=pred)).to_csv(dest / 'test_predictions.csv', index=False)
        report = classification_report(y,pred,target_names=test_ds.classes,output_dict=True)
        report['confusion_matrix'] = confusion_matrix(y,pred).tolist()
        (dest / 'test_metrics.json').write_text(json.dumps(dict(**row, classification_report=report), indent=2))
        print(json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT / 'results/comparison.csv', index=False)
    model, config, device = load_classifier(ROOT / 'checkpoints' / f'{best}.pt')
    custom = []
    for path in sorted((ROOT / 'custom_images').glob('*')):
        if path.suffix.lower() in {'.jpg','.jpeg','.png'}:
            custom.append(dict(image=path.name, **predict_image(path, model, config, device, tta=selections[best]['tta'])))
    (ROOT / 'results/custom_predictions.json').write_text(json.dumps(dict(model=best,predictions=custom), indent=2))


if __name__ == '__main__':
    main()
