## src folder

In this _src_ folder, we can find all the required scripts so as to perform the segmentation task. The folders _models_ and _preprocessing_, which contain their own _readme.md_ files, store, respectively, the different neural networks implemented and the instructions to convert the dataset into TFRecord files and viceversa. On the other hand, _utils_ consists on a set of functions defined by us used to perform some auxiliary tasks. Finally, two more scripts, _dataset.py_ and _train.py_, carry out the segmentation task. Both are explained below.

### dataset.py

The main goal of this script is to create the dataset we are going to work with. Notice that it could be a training, validation or testing dataset. In case of dealing with the first two, the output is going to be a pair of batches of the specified size (training or validation images and its corresponding labels). On the contrary, for the testing dataset it will only output the images batch.

When creating the dataset, we carry out a central crop so as to keep the 80% of the image. The reason is getting rid of the black space surrounding the brain, which does not provide any relevant information. Moreover, when generating the batches, we perform some shuffling so as to randomly pick the samples.


### train.py
