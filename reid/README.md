# CAL-ReID
This folder contains the implementation of the person re-identification experiments.

Our implementation is based on the code of [BoT](https://github.com/michuanhaohao/reid-strong-baseline).

## Requirements
- Python 3.6+
- PyTorch 1.4
- torchvision
- CUDA 10.1
- ignite=0.1.2(https://github.com/pytorch/ignite)
- yacs(https://github.com/rbgirshick/yacs)

## Prepare the data
- Download the Market1501, DukeMTMC-reID, and MSMT17 datasets.
- Change the path in `\data\datasets\dataset_name.py` to your dataset path.
- Change the path in `\configs\softmax_triplet.yml` to your model path.

## Reproduce the results
- Run script train.sh to train the model.
- Change the path in test.sh to the pre-trained model path and run `sh test.sh` for inference.
