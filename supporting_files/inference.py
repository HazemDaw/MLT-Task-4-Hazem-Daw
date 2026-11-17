"""Load a saved classifier and predict a custom RGB, grayscale or RGBA image."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageOps
import torch
from models import build_model
from train import preprocessing, amp


def load_classifier(checkpoint, device=None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = build_model(saved['config']['name'], pretrained=False)
    model.load_state_dict(saved['state_dict'])
    model.to(device).eval()
    return model, saved['config'], device


@torch.inference_mode()
def predict_image(image_path, model, config, device, tta=False):
    """Return the class, both class probabilities and confidence (not calibration)."""
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        x = preprocessing(config['name'], size=config['image_size'])(image).unsqueeze(0).to(device)
    model.eval()
    with amp(device):
        z = model(x)
        if tta:
            z = (z + model(x.flip(-1))) / 2
    p = z.float().sigmoid().item()
    names = {value: key for key, value in config['class_to_idx'].items()}
    index = int(p >= 0.5)
    return dict(predicted_class=names[index], confidence=p if index else 1 - p,
                probabilities={names[0]: 1 - p, names[1]: p})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--checkpoint', default='checkpoints/resnet18.pt')
    parser.add_argument('--device')
    parser.add_argument('--tta', action='store_true', help='Average original and horizontal-flip logits')
    args = parser.parse_args()
    model, config, device = load_classifier(args.checkpoint, args.device)
    print(json.dumps(predict_image(args.image, model, config, device, tta=args.tta), indent=2))
