# Least Squares Generative Adversarial Networks
Tensorflow implementation of [Least Squares Generative Adversarial Networks by Mao et al](https://arxiv.org/abs/1611.04076) (LSGAN).

## Prerequisites
- Python 2.7+
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [tqdm](https://pypi.python.org/pypi/tqdm)
- [Tensorflow r1.0+](https://www.tensorflow.org/install/)
- [lmdb](https://lmdb.readthedocs.io/en/release/) (for processing LSUN dataset only)


## Data
- [LSUN Scene Classification](http://lsun.cs.princeton.edu/)
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


## Preparation
1. Clone this repo, create `log/` and `ckpt/` folders:
   ```bash
   git clone https://github.com/markdtw/least-squares-gan.git
   cd least-squares-gan
   mkdir log
   mkdir ckpt
   ```
2. To train on LSUN, use the [provided tools](https://github.com/fyu/lsun) to download and extract. For example:
   ```bash
   python download.py -c conference_room
   unzip conference_room_train_lmdb.zip
   python data.py export conference_room_train_lmdb --out_dir conference_room_train_images --flat
   ```
   I replaced _.webp_ from [this line](https://github.com/fyu/lsun/blob/master/data.py#L49) to _.jpg_

3. To train on CelebA, I use [this file](https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py) to download. Shout out to carpedm20.

4. Now you are good to go, first time training on LSUN will center-crop all the images to 224x224 and store them in a new folder.

## Train
Train on LSUN conference room with default settings:
```bash
python main.py --train
```
Train on CelebA with default settings:
```bash
python main.py --train --dataset=CelebA
```
Train from a previous checkpoint at epoch X:
```bash
python main.py --train --modelpath=ckpt/lsgan-LSUN<CelebA>-X
```
Check out tunable hyper-parameters:
```bash
python main.py
```

## Some results

![tbg]()


## Notes
- The model will save 40 generated pictures in `log/` folder after every epoch.
- model.py is simply DCGAN, but the output of D is not the fc layer as mentioned in the paper, instead I use the last conv2d similar to [this repo](https://github.com/cameronfabbri/LSGANs-Tensorflow), I can't train it with the fc layer as the output layer because either D or G will be too strong that makes the gradients vanish. Please let me know if you find the problem.
- Issues are more than welcome!


## Resources
- [The paper](https://arxiv.org/abs/1611.04076)
- Highly based on [This repo](https://github.com/cameronfabbri/LSGANs-Tensorflow)

