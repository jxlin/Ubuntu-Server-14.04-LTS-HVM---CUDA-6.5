#!/usr/bin/env python
# -*- coding: utf-8 -*-


# coding: utf-8

# GraphLab Create allows you to get started with neural networks without being an expert by eliminating the need to choose a good architecture and hyper-parameter starting values. Based on the input data, the neuralnet_classifier.create() function chooses an  architecture to use and sets reasonable values for hyper-parameters. Let’s check this out on MNIST, a dataset composed of handwritten digits where the task is to identify the digit:

# In[1]:

data = graphlab.SFrame('http://s3.amazonaws.com/GraphLab-Datasets/mnist/sframe/train')
model = graphlab.neuralnet_classifier.create(data, target='label')


# Evaluating this model on the prediction data will tell us how well the model functioned:

# In[2]:

testing_data = graphlab.SFrame('http://s3.amazonaws.com/GraphLab-Datasets/mnist/sframe/test')
model.evaluate(testing_data)


# The training procedure breaks down something like this:

# * Stage 1: Train a DNN classifier on a large, general dataset. A good example is ImageNet ,with 1000 categories and 1.2 million images. GraphLab hosts a model trained on ImageNet to allow you to skip this step in your own implementation.
# * Stage 2: The outputs of each layer in the DNN can be viewed as a meaningful vector representaion of each image. Extract these feature vectors from the layer prior to the output layer on each image of your task.
# * Stage 3: Train a new classifier with those features as input for your own task.

# Stage 1 is re-usable for many different problems, and GraphLab is hosting the model so you don't have to train it yourself. Stage 2 is easy to do with GraphLab's API (as shown below), and Stage 3 is typically done with a simpler classifier than a deep learning model so it's easy to build yourself. In the end, this pipeline results in not needing to adjust hyper-parameters, faster training, and better performance even in cases where you don't have enough data to train a convention deep learning model. What's more, this technique is effective even if your Stage 3 classification task is relatively unrelated to the task Stage 1 is trained on.

# First, lets load in the model trained on ImageNet. This corresponds to the end of Stage 1 in our pipeline:

# In[3]:

pretrained_model = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')


# Now, let's load in the cats vs dogs images. We resize because the original ImageNet model was trained on 256 x 256 x 3 images:

# In[4]:

cats_dogs_sf = graphlab.SFrame('http://s3.amazonaws.com/GraphLab-Datasets/cats_vs_dogs/cats_dogs_sf')
cats_dogs_sf['image'] = graphlab.image_analysis.resize(cats_dogs_sf['image'], 256, 256, 3)


# And extract features, per Stage 2 of our pipeline:

# In[5]:

cats_dogs_sf['features'] = pretrained_model.extract_features(cats_dogs_sf)
cats_dogs_train, cats_dogs_test = cats_dogs_sf.random_split(0.8)


# And now, let's train a simple classifier as described by Stage 3

# In[6]:

simple_classifier = graphlab.classifier.create(cats_dogs_train, features = ['features'], target = 'label')


# And now, to see how our trained model did, we evaluate it:

# In[7]:

simple_classifier.evaluate(cats_dogs_test)


# For comparisons sake, let’s try using just the .create() method.

# In[8]:

model = graphlab.neuralnet_classifier.create(cats_dogs_train, target='label', features = ['image'] )
model.evaluate(cats_dogs_test)


# It’s always important to make sure any machine learning technique is consistent in its usefulness, and that its success is not afluke. In order to do that, I tested it on the CIFAR-10 dataset developed by Alex Krizhevsky. The CIFAR-10 dataset has 50000 training images and 10000 prediction images divided into 10 classes. Each images is of size 32x32.

# Let's repeat the procedure we just went through for the Cats vs Dogs dataset:

# In[9]:

cifar_train = graphlab.SFrame('http://s3.amazonaws.com/GraphLab-Datasets/cifar_10/cifar_10_train_sframe')
cifar_test = graphlab.SFrame('http://s3.amazonaws.com/GraphLab-Datasets/cifar_10/cifar_10_test_sframe')
# Preprocess
cifar_train['image'] = graphlab.image_analysis.resize(cifar_train['image'], 256, 256, 3)
cifar_test['image'] = graphlab.image_analysis.resize(cifar_test['image'], 256, 256, 3)
# Stage 2
cifar_train['features'] = pretrained_model.extract_features(cifar_train)
cifar_test['features'] = pretrained_model.extract_features(cifar_test)
# Stage 3
classifier = graphlab.classifier.create(cifar_train, features=['features'], target='label')
# Evaluate
classifier.evaluate(cifar_test)
