"""Reproducible training. Test images are not loaded by this module's training path."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms, models
from models import build_model

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache' / 'torch'))
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def preprocessing(name, training=False, size=128):
    if name == 'cnn':
        return transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()])
    if name in ('vgg16', 'resnet18'):
        weights = models.VGG16_Weights.IMAGENET1K_V1 if name == 'vgg16' else models.ResNet18_Weights.IMAGENET1K_V1
        return weights.transforms()
    if training:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15, 0.15, 0.15, 0.03),
            transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                transforms.Normalize((0.5,) * 3, (0.5,) * 3)])


def loader(dataset, batch_size=64, shuffle=False, workers=4, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))


def remove_training_duplicates(dataset, data_root):
    audit_path = ROOT / 'results/data_audit.json'
    if not audit_path.exists():
        raise FileNotFoundError('Run prepare_data.py before training; the duplicate audit is required.')
    audit = json.loads(audit_path.read_text())
    excluded = set()
    for group in audit['duplicate_groups']:
        training = sorted(p for p in group if p.startswith('train/'))
        has_evaluation = any(not p.startswith('train/') for p in group)
        excluded.update(training if has_evaluation else training[1:])
    dataset.samples = [(p, y) for p, y in dataset.samples
                       if Path(p).relative_to(data_root).as_posix() not in excluded]
    dataset.imgs = dataset.samples
    dataset.targets = [y for _, y in dataset.samples]
    return sorted(excluded)


def amp(device):
    return torch.autocast(device_type='cuda', dtype=torch.float16) if device.type == 'cuda' else contextlib.nullcontext()


@torch.inference_mode()
def evaluate(model, batches, device, tta=False):
    model.eval()
    ys, logits = [], []
    for x, y in batches:
        x = x.to(device, non_blocking=True).float()
        with amp(device):
            z = model(x)
            if tta:
                z = (z + model(x.flip(-1))) / 2
        logits.append(z.float().cpu())
        ys.append(y.cpu())
    y, z = torch.cat(ys).float(), torch.cat(logits)
    return dict(loss=nn.functional.binary_cross_entropy_with_logits(z, y).item(),
                accuracy=((z >= 0) == y.bool()).float().mean().item(),
                y=y.numpy(), probabilities=z.sigmoid().numpy())


def extract_features(model, dataset, name, split, device, workers, batch_size):
    cache = ROOT / '.cache' / f'{name}_{split}'
    cache.parent.mkdir(exist_ok=True, parents=True)
    # Cache is tied to exact image bytes/order and preprocessing implementation.
    signature = hashlib.sha256(Path(__file__).read_bytes())
    for path, label in dataset.samples:
        signature.update(Path(path).read_bytes())
        signature.update(str(label).encode())
    digest = signature.hexdigest()
    marker = cache.with_suffix('.json')
    if marker.exists() and json.loads(marker.read_text())['sha256'] == digest:
        return np.load(cache.with_suffix('.npy'), mmap_mode='r')
    batches = loader(dataset, batch_size=batch_size, workers=workers)
    model.eval()
    features, offset = None, 0
    with torch.inference_mode():
        for x, _ in batches:
            with amp(device):
                f = model.features(x.to(device)).cpu().numpy().astype(np.float16)
            if features is None:
                features = np.lib.format.open_memmap(cache.with_suffix('.npy'), mode='w+',
                                                     dtype='float16', shape=(len(dataset), f.shape[1]))
            features[offset:offset + len(f)] = f
            offset += len(f)
            if offset % (batch_size * 50) == 0:
                print(f'{name}: extracted {split} {offset}/{len(dataset)}', flush=True)
    features.flush()
    marker.write_text(json.dumps(dict(sha256=digest, count=offset)))
    return np.load(cache.with_suffix('.npy'), mmap_mode='r')


class FeatureDataset(Dataset):
    def __init__(self, features, targets):
        self.features, self.targets = features, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.from_numpy(self.features[index].astype(np.float32)), self.targets[index]


class HeadWrapper(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x):
        return self.head(x).squeeze(1)


def train_run(name, epochs, seed=42, size=128, batch_size=64, workers=4, lr=None,
              data_root=None, run_name=None):
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(data_root or ROOT / 'data' / 'Cats_and_Dogs')
    run_name = run_name or name
    output = ROOT / 'results' / run_name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ROOT / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    train_ds = datasets.ImageFolder(data_root / 'train', preprocessing(name, True, size))
    excluded = remove_training_duplicates(train_ds, data_root)
    val_ds = datasets.ImageFolder(data_root / 'val', preprocessing(name, False, size))
    assert train_ds.class_to_idx == val_ds.class_to_idx
    assert list(train_ds.class_to_idx.values()) == [0, 1]
    pretrained = name in ('vgg16', 'resnet18')
    model = build_model(name, pretrained=pretrained).to(device)
    actual_size = 150 if name == 'cnn' else 224 if pretrained else size
    config = dict(name=name, run_name=run_name, epochs=epochs, seed=seed, image_size=actual_size,
                  batch_size=batch_size, class_to_idx=train_ds.class_to_idx,
                  pretrained=pretrained, train_images=len(train_ds), validation_images=len(val_ds),
                  excluded_training_duplicates=excluded,
                  device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                  torch=str(torch.__version__), selection='minimum validation BCE, threshold fixed at 0.5',
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if pretrained:
        train_f = extract_features(model, train_ds, name, 'train', device, workers, batch_size)
        val_f = extract_features(model, val_ds, name, 'val', device, workers, batch_size)
        train_batches = loader(FeatureDataset(train_f, train_ds.targets), batch_size=256, shuffle=True, workers=0, seed=seed)
        val_batches = loader(FeatureDataset(val_f, val_ds.targets), batch_size=256, workers=0)
        fit_model = HeadWrapper(model.head)
    else:
        train_batches = loader(train_ds, batch_size, True, workers, seed)
        val_batches = loader(val_ds, batch_size, False, workers, seed)
        fit_model = model
    lr = lr or {'cnn': 0.001, 'vgg16': 0.0001, 'resnet18': 0.001, 'scratch': 0.002}[name]
    config['lr'] = lr
    config['weight_decay'] = 0.0001 if name == 'scratch' else 0.0
    config['augmentation'] = 'random crop, horizontal flip, color jitter; 0.05 label smoothing' if name == 'scratch' else 'none'
    optimizer = torch.optim.AdamW((p for p in fit_model.parameters() if p.requires_grad), lr=lr,
                                   weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01) if name == 'scratch' else None
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best, history, started = math.inf, [], time.perf_counter()
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    for epoch in range(1, epochs + 1):
        tic = time.perf_counter()
        fit_model.train()
        total_loss, correct, total = 0.0, 0, 0
        epoch_lr = optimizer.param_groups[0]['lr']
        for x, y in train_batches:
            x, y = x.to(device, non_blocking=True).float(), y.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            targets = y * 0.95 + 0.025 if name == 'scratch' else y
            with amp(device):
                z = fit_model(x)
                loss = nn.functional.binary_cross_entropy_with_logits(z, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(fit_model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y)
            correct += ((z >= 0) == y.bool()).sum().item()
            total += len(y)
        val = evaluate(fit_model, val_batches, device)
        row = dict(epoch=epoch, train_loss=total_loss / total, train_accuracy=correct / total,
                   val_loss=val['loss'], val_accuracy=val['accuracy'], learning_rate=epoch_lr,
                   seconds=time.perf_counter() - tic)
        history.append(row)
        if val['loss'] < best:
            best = val['loss']
            torch.save(dict(state_dict=model.state_dict(), config=config, best_epoch=epoch,
                            validation_loss=best, validation_accuracy=val['accuracy']),
                       checkpoint_dir / f'{run_name}.pt')
        with (output / 'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"{run_name} {epoch}/{epochs}: train {row['train_accuracy']:.4f}, val {row['val_accuracy']:.4f}, val BCE {val['loss']:.4f}, {row['seconds']:.1f}s", flush=True)
        if scheduler:
            scheduler.step()
    config['training_seconds'] = time.perf_counter() - started
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['cnn', 'vgg16', 'resnet18', 'scratch'], required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--run-name')
    args = parser.parse_args()
    train_run(args.model, args.epochs, args.seed, args.size, args.batch_size, args.workers,
              args.lr, run_name=args.run_name)
