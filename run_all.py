"""Reproduce all experiments in a fixed order, then unlock test evaluation."""
from train import train_run


if __name__ == '__main__':
    train_run('cnn', epochs=15, batch_size=128)
    train_run('vgg16', epochs=15, batch_size=64)
    train_run('resnet18', epochs=15, batch_size=64)
    train_run('scratch', epochs=60, size=128, batch_size=128)
    from evaluate import main
    main()
    from build_artifacts import build_figures
    build_figures()
