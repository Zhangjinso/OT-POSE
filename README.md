# Optimal-Transport GAN

## Installation

#### Requirements

- Python 3
- PyTorch 1.7.1
- CUDA 10.2

#### Conda Installation

``` bash
# 1. Create a conda virtual environment.
conda create -n OTGAN python=3.6
conda activate OTGAN
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2

# 2. Install other dependencies
pip install -r requirements.txt
```

## Dataset

- Download `img_highres.zip` of the DeepFashion Dataset from [In-shop Clothes Retrieval Benchmark](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00). 

- Unzip `img_highres.zip`. You will need to ask for password from the [dataset maintainers](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). Then rename the obtained folder as **img** and put it under the `./dataset/deepfashion` directory. 

- We split the train/test set following [GFLA][https://github.com/RenYurui/Global-Flow-Local-Attention]. Several images with significant occlusions are removed from the training set. Download the train/test pairs and the keypoints `pose.zip` extracted with [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) by runing: 

  ```bash
  cd scripts
  ./download_dataset.sh
  ```

  Or you can download these files manually：

  - Download the train/test pairs from [Google Drive](https://drive.google.com/drive/folders/1PhnaFNg9zxMZM-ccJAzLIt2iqWFRzXSw?usp=sharing) including **train_pairs.txt**, **test_pairs.txt**, **train.lst**, **test.lst**. Put these files under the  `./dataset/deepfashion` directory. 
  - Download the keypoints `pose.zip` extracted with Openpose from [Google Driven](https://drive.google.com/file/d/1HXu7LDLW45Aw0n3a1W9HuUXARCfXAU7R/view?usp=sharing). Unzip and put the obtained floder under the  `./dataset/deepfashion` directory.

- Run the following code to save images to lmdb dataset.

  ```bash
  python -m scripts.prepare_data \
  --root ./dataset/deepfashion \
  --out ./dataset/deepfashion
  ```



## Training 

This project supports multi-GPUs training. The following code shows an example for training the model with $256 \times 176$ images using 2 GPUs.

  ```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 \
--master_port 1234 train.py \
--config ./config/fashion_256.yaml \
--name $name_of_your_experiment
  ```

All configs for this experiment are saved in `./config/fashion_256.yaml`. 
If you change the number of GPUs, you may need to modify the batch_size in `./config/fashion_256.yaml` to ensure using a same batch_size.

## Inference

- **Download the trained weights for [256x176 images](https://drive.google.com/drive/folders/1aP6rU4pBj-_ZCdBhg4HgtyUMHZXdEphn?usp=sharing)**. Put the obtained checkpoints under  `./result/fashion_256` respectively.

- Run the following code to evaluate the trained model:

  ```bash
  python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port 12359 inference.py \
  --config ./config/fashion_256.yaml \
  --name fashion_256 \
  --which_iter 215881
  --no_resume \
  --output_dir ./result/fashion_256/inference 
  ```

The result images are save in  `./result/fashion_256/inference `. 



## Evaluation

The evaluation process follows [repo](https://github.com/Zhangjinso/Pose_Transfer_Evaluation)

