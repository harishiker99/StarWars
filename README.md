# Classification of custom dataset using fastai library and PyTorch 1.5

# Transfer Learning
Transfer learning (TL) is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.

![]("https://github.com/harishiker99/StarWars/blob/master/screenshots/Capture5.jpg")
<strong>No. of classes: 2 <br>
  Name of the classes: <ol><li>Sith</li><li>Jedi</li></ol></strong>
  
<img src="https://github.com/harishiker99/StarWars/blob/master/screenshots/Capture.JPG">

# Convolutional Neural Network
A CNN consists of a number of convolutional and subsampling layers optionally followed by fully connected layers. 
The input to a convolutional layer is a m x m x r image where m is the height and width of the image and r is the number of channels, e.g. an RGB image has r=3. The convolutional layer will have k filters (or kernels) of size n x n x q where n is smaller than the dimension of the image and q can either be the same as the number of channels r or smaller and may vary for each kernel. The size of the filters gives rise to the locally connected structure which are each convolved with the image to produce k feature maps of size m−n+1. Each map is then subsampled typically with mean or max pooling over p x p contiguous regions where p ranges between 2 for small images (e.g. MNIST) and is usually not more than 5 for larger inputs. Either before or after the subsampling layer an additive bias and sigmoidal nonlinearity is applied to each feature map. 
The figure below illustrates a full layer in a CNN consisting of convolutional and subsampling sublayers. They consist of shared weights.
After the convolutional layers there may be any number of fully connected layers. The densely connected layers are identical to the layers in a standard multilayer neural network.

<img src="https://github.com/harishiker99/StarWars/blob/master/screenshots/Capture4.jpg">

# Architecture used: ResNet-34
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
Resnet models were proposed in “Deep Residual Learning for Image Recognition”. Here we have the 5 versions of resnet models, which contains 5, 34, 50, 101, 152 layers respectively. Detailed model architectures can be found in Table 1. Their 1-crop error rates on imagenet dataset with pretrained models are listed below.

<img src="https://github.com/harishiker99/StarWars/blob/master/screenshots/Capture6.JPG">

# Learning rate: a range of values from 0.00001 to 0.001

<img src="https://github.com/harishiker99/StarWars/blob/master/screenshots/Capture3.JPG">

# Result: 85% accuracy over 10 epochs

<img src="https://github.com/harishiker99/StarWars/blob/master/screenshots/Capture2.JPG">
