"""Cross-check reported scores against per-image predictions and saved histories."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parent


def main():
    table = pd.read_csv(ROOT/'results/comparison.csv')
    assert set(table.model) == {'cnn','vgg16','resnet18','scratch'}
    selection = json.loads((ROOT/'results/selection.json').read_text())
    checks = []
    for _, row in table.iterrows():
        folder = ROOT/'results'/row.model
        config = json.loads((folder/'config.json').read_text())
        history = pd.read_csv(folder/'history.csv')
        predictions = pd.read_csv(folder/'test_predictions.csv')
        assert len(history) == config['epochs']
        assert np.isfinite(history.select_dtypes('number').to_numpy()).all()
        assert len(predictions) == 2500
        assert predictions.target.value_counts().to_dict() == {0:1250,1:1250}
        assert predictions.probability_dog.between(0,1).all()
        assert np.array_equal((predictions.probability_dog>=0.5).astype(int),predictions.prediction)
        accuracy = (predictions.target==predictions.prediction).mean()
        assert abs(accuracy-row.accuracy) < 1e-10
        assert int((predictions.target==predictions.prediction).sum()) == row.correct
        expected_epoch = int(history.loc[history.val_loss.idxmin(),'epoch'])
        assert row.best_epoch == expected_epoch
        assert config['pretrained'] == (row.model in {'vgg16','resnet18'})
        assert config['train_images'] == 19995
        digest=hashlib.sha256((ROOT/'checkpoints'/f'{row.model}.pt').read_bytes()).hexdigest()
        assert digest==selection['models'][row.model]['checkpoint_sha256']
        checks.append(f'{row.model}: history, validation selection, 2500 predictions and checkpoint verified')
    manifest = pd.read_csv(ROOT/'results/data_manifest.csv')
    excluded=json.loads((ROOT/'results/cnn/config.json').read_text())['excluded_training_duplicates']
    training=manifest[(manifest.split=='train')&(~manifest.path.isin(excluded))]
    held=manifest[manifest.split!='train']
    assert len(training)==19995
    assert not set(training.pixel_sha256)&set(held.pixel_sha256)
    assert training.pixel_sha256.is_unique
    checks.append('No exact decoded-pixel duplicates within retained training or against validation/test')
    notebook=nbformat.read(ROOT/'CNN_and_Transfer_Learning_Pytorch.ipynb',as_version=4)
    nbformat.validate(notebook)
    cells=[c for c in notebook.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in cells)
    assert not any(o.output_type=='error' for c in cells for o in c.outputs)
    checks.append(f'Notebook schema valid; {len(cells)} code cells executed without errors')
    result=dict(status='passed',checks=checks)
    (ROOT/'results/verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
