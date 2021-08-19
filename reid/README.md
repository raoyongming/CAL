# CAL-ReID
This folder contains the implementation of the person re-identification experiments.

Our implementation is based on the code of [BoT](https://github.com/michuanhaohao/reid-strong-baseline).

## Prepare the data
Download the Market1501 dataset from [http://www.liangzheng.org/Project/project_reid.html](http://www.liangzheng.org/Project/project_reid.html)

## Reproduce the results
- Change the path in `\data\datasets\market1501.py` to your path of the Market1501 dataset
- Change the path in `\configs\softmax_triplet.yml` to your path to save models.
- Run script train.sh to reproduce the results

## Requirements
`pip install -r requirements.txt`


