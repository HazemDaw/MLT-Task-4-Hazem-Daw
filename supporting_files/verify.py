"""Focused checks for output shapes, gradient flow, frozen BatchNorm and inference."""
import tempfile
from pathlib import Path
import unittest
from PIL import Image
import numpy as np
import torch
from torch import nn
from models import CNNModel, ScratchResNet, TransferModel
from inference import predict_image
from train import evaluate, loader, seed_everything


class ClassifierChecks(unittest.TestCase):
    def setUp(self):
        seed_everything(42)

    def test_baseline_has_three_convolutions_and_learns(self):
        model = CNNModel()
        self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in model.modules()), 3)
        z = model(torch.randn(3, 3, 150, 150))
        self.assertEqual(z.shape, (3,))
        nn.functional.binary_cross_entropy_with_logits(z, torch.tensor([0., 1., 0.])).backward()
        self.assertTrue(torch.isfinite(model.features[0].weight.grad).all())
        self.assertGreater(model.features[0].weight.grad.abs().sum().item(), 0)

    def test_scratch_has_trainable_random_features(self):
        model = ScratchResNet()
        z = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(z.shape, (2,))
        z.sum().backward()
        self.assertIsNotNone(model.features[0].weight.grad)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_frozen_features_do_not_change_batchnorm(self):
        model = TransferModel('resnet18', pretrained=False)
        before = {k: v.clone() for k, v in model.features.state_dict().items()}
        model.train()
        z = model(torch.randn(2, 3, 64, 64))
        z.sum().backward()
        self.assertTrue(all(p.grad is None for p in model.features.parameters()))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.features.state_dict().items()))
        self.assertIsNotNone(model.head[0].weight.grad)

    def test_evaluation_weights_partial_batch_correctly(self):
        model = nn.Identity()
        x = torch.tensor([-2., 2., 0.5, -1., -0.5])
        y = torch.tensor([0, 1, 1, 0, 1])
        batches = loader(torch.utils.data.TensorDataset(x, y), batch_size=3, workers=0)
        result = evaluate(model, batches, torch.device('cpu'))
        expected = nn.functional.binary_cross_entropy_with_logits(x, y.float()).item()
        self.assertAlmostEqual(result['loss'], expected, places=6)
        self.assertAlmostEqual(result['accuracy'], 0.8, places=6)

    def test_custom_image_converts_grayscale_and_uses_class_map(self):
        class Constant(nn.Module):
            def forward(self, x):
                assert x.shape == (1, 3, 150, 150)
                return torch.zeros(len(x)) - 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gray.png'
            Image.new('L', (20, 30)).save(path)
            result = predict_image(path, Constant(), dict(name='cnn', image_size=150,
                                   class_to_idx={'cats': 0, 'dogs': 1}), torch.device('cpu'))
        self.assertEqual(result['predicted_class'], 'cats')
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
