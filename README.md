## CAM_MatConvNet

This is a simple implementation of the discriminative localization method in [1]. We also made some changes in order to better fit the Pascal VOC dataset.

I) We employed a multi-label loss in [2].

II) We added an additional layer after 3x3x1024 for better and easy visualization. The size of layer is (1,1,20), which means that every channel can be regarded as a visualization of the corresponding class.

### How to run it ?
#### For training a new model
1. Download vgg-16 from the homepage of MatConvNet [this link](http://www.vlfeat.org/matconvnet/pretrained/), and put it in vgg_net/.

2. Download imdb.mat [this link](https://mega.nz/#!h5AkjbKC). Some suggestions: This imdb file combines pascal voc 2007 & 2012. The train set contains trainval_2007, test_2007, and train_2012, while the test set contains val_2012. We marked positive class as 1 while the negative one is -1. You might feel free to make another imdb file by yourself by modifying get_batch function in CAM_MatConvNet.m.

3. Compile matconvnet and then run CAM_MatConvNet to train a new model. (For more details, please refer to the tutorial of MatConvNet).

4. Run visualization.m to visualize all images in imdb.mat

#### For using a pretrained model
1. You could download the pretrained model [here](https://mega.nz/#!I0onxLpD), and then put it in data/.

2. Run visualization.m


[1] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba. Learning Deep Features for Discriminative Localization. CVPR'16

[2] MaxiMe Oquab. Is object localization for free? – Weakly-supervised learning with convolutional neural networks. CVPR'15
