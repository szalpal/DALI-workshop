This repository comprises code for DALI workshop

docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/dali -w /dali nvcr.io/nvidia/pytorch:25.01-py3 python rn50.py --data-dir /dali/train_data/train
