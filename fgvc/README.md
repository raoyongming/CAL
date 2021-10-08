# CAL-FGVC
This folder contains the implementation of the fine-grained image classification experiments.

Our implementation is based on the Pytorch version code of [WS-DAN](https://github.com/GuYuc/WS-DAN.PyTorch).

## Prepare the data

### CUB
Download CUB-200-2011 dataset from [this link](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and move the uncompressed data folder to `./CUB-200-2011`. The data structure should be:

  ```
  ./CUB-200-2011
          └─── images.txt
          └─── image_class_labels.txt
          └─── train_test_split.txt
          └─── images
                  └─── 001.Black_footed_Albatross
                          └─── Black_Footed_Albatross_0001_796111.jpg
                          └─── ...
                  └─── 002.Laysan_Albatross
                  └─── ...
  ```
### Stanford Cars
Download Stanford Cars dataset from [this link](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) and move the uncompressed data folder to `./stanford_cars`. The data structure should be: 

  ```
  -/stanford_cars
        └─── car_ims
                  └─── 00001.jpg
                  └─── 00002.jpg
                  └─── ...
        └─── cars_annos.mat
  ```

### FGVC-Aircraft
Download FGVC-Aircraft dataset from [this like](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) and move the uncompressed data folder to `./fgvc-aircraft-2013b`. The data structure should be: 

  ```
  ./fgvc-aircraft-2013b/data/
                  └─── images
                          └─── 0034309.jpg
                          └─── 0034958.jpg
                          └─── ...
                  └─── variants.txt
                  └─── images_variant_trainval.txt
                  └─── images_variant_test.txt
  ```

## Training & Evaluation
- Modify `config_distributed.py` to run experiments on different datasets
- Run `bash train_distributed.sh` to train models.
- Set configurations in ```config_infer.py``` and run  `python infer.py` to conduct multi-crop evaluation.

## Requirements
* Python 3
* PyTorch 1.0+
* Apex


