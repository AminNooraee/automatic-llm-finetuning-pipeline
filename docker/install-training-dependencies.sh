#!/bin/sh
set -eu

python -m pip install --index-url "${PYTORCH_INDEX_URL}" \
    -r docker/pytorch-requirements.txt
python -c "from importlib.metadata import version; print('\n'.join(name + '==' + version(name) for name in ('torch', 'torchvision', 'torchaudio')))" \
    > /opt/pytorch-constraints.txt
python -m pip install --constraint /opt/pytorch-constraints.txt \
    -r requirements.txt
python -m pip check
python -m pip freeze > /opt/deployment-requirements.txt
python -c "import torch, torchvision, torchaudio, llamafactory; from fine_tuning_pipeline.train_pipeline import load_config; assert load_config()['model']['name']; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda)"
