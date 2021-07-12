# Multi-output Gaussian Processes with Augment & Reduce 

This repository contains the implementation of our Multi-output Gaussian Processes with Augment & Reduce (MOGPs-AR) model. The entire code is written in Python and highly depends on GPflow package.


Traditional multi-output Gaussian processes (MOGPs) for multi-class classification has been less studied. The reason is twofold: 1) when using a softmax function, it is not clear how to scale it beyond the case of a few outputs; 2) most common type of data in multi-class classification problems consists of image data, and MOGPs are not specifically designed to handle such high-dimensional data.

We introduce a new extension of multi-output Gaussian processes, called Multi-output Gaussian Processes with Augment & Reduce (MOGPs-AR),able to handle large scale multi-output multi-class classification problems, typically in the range of hundreds and even thousands of classes. We also enable our model to efficiently deal with such high dimensional input such as images by employing convolutional kernels.


The model is implemented based on [GPflow 2.1.3](https://github.com/GPflow/GPflow) or [GPflow 2.1.4](https://github.com/GPflow/GPflow).

One example is provided in 

MOGPs-AR_balance-scale.data.ipynb

Other examples see the folder Experiment_demo.
