# CSC411 - Machine Learning, Assignment 3
This is the GitHub repository for the third assignment of CSC411.

## Authors
Jason Li
<br>
Nathan Kong

## Dependencies: sklearn, skimage, tensorflow, keras, protobuf

To install sklearn and skimage:
<br>
~~~~
sudo pip install scikit-learn --upgrade
sudo pip install skimage --user --upgrade
sudo pip install keras
~~~~
<br>
Try installing without the options first. The installation can fail if their dependencies are not installed.
<br>
If so, follow the instructions to install the dependencies.

It also uses the VGG16 pretrained model form https://github.com/fchollet/deep-learning-models
I also made some changes to VGG16.py for testing but it can be safely ignored.

## How to run
~~~~
python main.py
~~~~

## Reference
VGG16: <a href="https://arxiv.org/abs/1409.1556">Very Deep Convolutional Networks for Large-Scale Image Recognition</a>

