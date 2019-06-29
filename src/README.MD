In this _src_ folder, we can find all the required scripts so as to perform the segmentation task. The folders _models_ and _preprocessing_, which contain their own _readme.md_ files, store, respectively, the different neural networks implemented and the instructions to convert the dataset into TFRecord files and viceversa. On the other hand, _utils_ consists on a set of functions defined by us used to perform some auxiliary tasks. Finally, three more scripts, _dataset.py_, _train.py_ and _main.py_, carry out the segmentation task. They are explained below.

## dataset.py
It contains different functions regarding the dataset.

##### get_tensor_size

It gives the number of input and output channels of the labels according to the desired codification: one hot encoding and/or binarization.

##### get_file_lists

This function returns three lists (train_list, valid_list and test_list) containing the directories of all the files in the specified path. The parameter _data_dir_ gives the directory where the TFRecords are stored.

##### create_dataset
The main goal of this script is to create the dataset we are going to work with. Notice that it could be a training, validation or testing dataset. In case of dealing with the first two, the output is going to be a pair of batches of the specified size (training or validation images and its corresponding labels). On the contrary, for the testing dataset it will only output the images batch. \
When creating the dataset, we carry out a central crop so as to keep the 80% of the image. The reason is getting rid of the black space surrounding the brain, which does not provide any relevant information. Moreover, when generating the batches, we perform some shuffling so as to randomly pick the samples.
It is worth noting that the construction of the dataset is customized by means of two different parameters: _perform_one_hot_ and _binarize_labels_. The former says whether we want the labels expressed using the one_hot encoding and the latter gives the option of binarizing the labels. This last part implies, as already mentioned in the README file of the repository, performing a binary classification between background and tumour.   


## train.py


## main.py

This script is the one that we have to run so as to execute the _train.py_. In it, we have defined the different hyperparameters used along the already mentioned script. However, it is also possible to customize such values when calling the _main.py_. See the _argparse_ part so as to see how to introduce the different parameters.
