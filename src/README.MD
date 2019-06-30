In this _src_ folder, we can find all the required scripts so as to perform the segmentation task. The folders _models_ and _preprocessing_, which contain their own README files, store, respectively, the different neural networks implemented and the instructions to convert the dataset into TFRecord files and viceversa. On the other hand, _utils_ consists on a set of functions defined by us used to perform some auxiliary tasks. Finally, three more scripts, _dataset.py_, _train.py_ and _main.py_, carry out the segmentation task. In addition, _test.py_ performs the testing. They are explained below.

## dataset.py
It contains different functions regarding the dataset.

#### get_tensor_size

It gives the number of input and output channels of the labels according to the desired codification: one hot encoding and/or binarization.

#### get_file_lists

This function returns three lists (train_list, valid_list and test_list) containing the directories of all the files in the specified path. The parameter _data_dir_ gives the directory where the TFRecords are stored.

#### create_dataset
The main goal of this script is to create the dataset we are going to work with. Notice that it could be a training, validation or testing dataset. In case of dealing with the first two, the output is going to be a pair of batches of the specified size (training or validation images and its corresponding labels). On the contrary, for the testing dataset it will only output the images batch. \
When creating the dataset, we carry out a central crop so as to keep the 80% of the image. The reason is getting rid of the black space surrounding the brain, which does not provide any relevant information. Moreover, when generating the batches, we perform some shuffling so as to randomly pick the samples.\
It is worth noting that the construction of the dataset is customized by means of two different parameters: _perform_one_hot_ and _binarize_labels_. The former says whether we want the labels expressed using the one_hot encoding and the latter gives the option of binarizing the labels. This last part implies, as already mentioned in the README file of the repository, performing a binary classification between background and tumour.   


## train.py

This function performs the segmentation itself. Let us now develop the different steps.

First of all, it creates one iterator for each of the two datsets of training and validation (_train_iterator_ and _validation_iterator_) along with a feedable element so as to switch between the previous two (_handle_). Combining these ingredients, we construct an iterator, _iterator_, that gives a batch of the desired dataset in such a way that it recalls the previous generated batch and starts from this point. That is, it does not give the same batch all the time. It is worth noting the fact that the images and the labels have to be floats (they cannot be integers) so as to perform the convolutions of the U-Net architecture.

As we already mentioned in the README of the repository, we take different approaches on the data: performing one-hot encoding and/or binarizing the labels. This information is input by means of two parameters, _perform_one_hot_ and _binarize_labels_. However, at the begining of the _main_ function, we translate this information into two other variables that are _label_input_size_ and _label_output_size_. According to the values of this variables, the one-hot encoding and the binarization are going to be implemented or not. The last table inside the comment at the begining of the function depicts this idea.

Once the output of the model has been computed, we compute the losses and the metrics. The chosen loss will depend on the _label_input_size_ and _label_output_size_, as explained at the beginning of the comment. On the other hand, so as to compute the IoU metrics, the labels and logits have to be one-hot encoded. Therefore, such metrics can only be implemented in case _label_input_size>1_.

Afterwards, we perform an optimization step using the Adam optimizer, which uses a default learning rate of 1e-5. Finally, we use the _tf.summary_ function so as to visualize the results on tensorboard.

At this point, we run the session. We begin by inicializing the training iterator (_train_iterator_) and then, we start computing the loss for each batch. Every _step_metrics_ batches of train images, we perform a validation epoch. That is, we compute the loss and IoU metrics for each validation batch. Their means are going to be the loss and metrics associated to the whole epoch. Observe that at the beginning of each epoch we are initializing the validation iterator (_validation_iterator_).\
In addition, every _steps_saver_ batches of training images, we save the weights.


## main.py

This script is the one that we have to run so as to execute the _train.py_. In it, we have defined the different hyperparameters used along the already mentioned script. However, it is also possible to customize such values when calling the _main.py_. See the _argparse_ part so as to see how to introduce the different parameters.
