"""Execute the review notebook and export HTML, including a Windows DLL workaround."""
import os
import sys
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    runtime = root / '.cache' / 'jupyter'
    for name in ['config', 'runtime', 'ipython', 'data/kernels/mlt-task4']:
        (runtime / name).mkdir(parents=True, exist_ok=True)
    os.environ['JUPYTER_CONFIG_DIR'] = str(runtime / 'config')
    os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime / 'runtime')
    os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
    os.environ['JUPYTER_PATH'] = os.pathsep.join([str(runtime / 'data'), str(Path(sys.base_prefix) / 'share/jupyter')])
    (runtime / 'data/kernels/mlt-task4/kernel.json').write_text(json.dumps(dict(
        argv=[sys.executable, str(Path(__file__).resolve()), '--kernel', '-f', '{connection_file}'],
        display_name='MLT Task 4', language='python')))
    import nbformat
    from nbclient import NotebookClient
    notebook = root / 'CNN_and_Transfer_Learning_Pytorch.ipynb'
    nb = nbformat.read(notebook, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name='mlt-task4', resources={'metadata': {'path': str(root)}}).execute()
    nbformat.validate(nb)
    nbformat.write(nb, notebook)
    print('Notebook executed without errors.', flush=True)
    from nbconvert import HTMLExporter
    body, _ = HTMLExporter().from_notebook_node(nb)
    (root / 'Task_4_Notebook.html').write_text(body, encoding='utf-8')
    print('HTML review copy exported.', flush=True)


if __name__ == '__main__':
    if '--kernel' in sys.argv:
        sys.argv.remove('--kernel')
        # Load PyTorch before the kernel's native extensions to avoid Windows DLL conflicts.
        import torch
        from ipykernel.kernelapp import IPKernelApp
        IPKernelApp.launch_instance()
    else:
        main()
