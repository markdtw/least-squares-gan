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
1. Clone this repo, create `ckpt/` folder:
   ```bash
   git clone https://github.com/markdtw/least-squares-gan.git
   cd least-squares-gan
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

Epoch 10:
![ep-10](https://github.com/markdtw/least-squares-gan/blob/master/log/generated-ep-10.jpg)

Epoch 25:
![ep-25](https://github.com/markdtw/least-squares-gan/blob/master/log/generated-ep-25.jpg)

Epoch 45:
![ep-45](https://github.com/markdtw/least-squares-gan/blob/master/log/generated-ep-45.jpg)

Results from epoch 45 is already nice and crispy.

Generator loss:
![g-loss](https://github.com/markdtw/least-squares-gan/blob/master/log/gloss.png)

Discriminator loss:
![d-loss](https://github.com/markdtw/least-squares-gan/blob/master/log/dloss.png)


## Notes
- The model will save 40 generated pictures in `log/` folder every epoch.
- Initialization is important! Default initialization with `tf.xavier_initializer` will lead to either D or G's gradient vanishing problem, instead I use `tf.truncated_normal_initializer` which is identical to DCGAN original implementation to solve the problem.
- Issues are more than welcome!


## Resources
- [The paper](https://arxiv.org/abs/1611.04076)
- Highly based on [This repo](https://github.com/cameronfabbri/LSGANs-Tensorflow)

