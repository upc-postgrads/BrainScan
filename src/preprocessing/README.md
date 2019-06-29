### GenerateTFRecords.py
As we already said in the README file of the repository, the dataset that we used was the one from the BRATS challenge. In such dataset, each patient is represented as a nifti file. Therefore, in this script we read each of these nifti files and store them as a TFRecord by means of the function _tf.data.TFRecords.Dataset()_.\
Recall that in the already mentioned README, we explained how to represent each patient as a concatenation of slices. That was the procedure followed so as to perform the storage of the dataset.

In the script, we begin by defining some functions that allow us to:<br />
1- Store the ID of every patient<br />
2- Discard fully black images that would add noise to the training<br />
3- Normalize and also store the valid images in grayscale<br />
<br />
After having defined these helper blocks, then we execute the main function, which does the following:
<br /><br />
1- First, it defines different "writers" with tf.python_io.TFRecordWriter for 3 different batches, which are training, validation and testing.<br /><br />
2- Then we use the library Nibabel which is useful for transforming .NIFTI images into raw data.<br /><br />
3- We slice this raw 3D data in smaller 2D images and we do it with the 4 volumes from only 1 patient, so that we end up with 155 * 4 images<br /><br />
4- These images are parsed to become grayscale, discarted if they are fully black and stored in order in a list, which is later on converted into a nummpy array<br /><br />
5- We convert the labels into np.uint8<br /><br />
6- After that we create a dictionary of keys and values from the attributes of this patient: image, label, PatientID,...etc with the tf.train.Example function<br /><br />
7- Finally we convert some of the values of the dictionary from a list of bytes and int.64 to a tensor of bytes and int.64 respectively, with the tf.train.Feature(). The reason for this is that the tf.data.TFRecordDataset() function needs to receive data with this structure of tensors to work within the tensorflow framework<br /><br />
8- Finally, everything is done in 3 differents loops, one for every batch type (training, validation and test), that parse all the patients.<br /><br />
9- At the end the batch size of the training and validation sets respecitvely, will depend on the % of patients that we want to keep for validating purposes<br /><br />
10- This generator will be called everytime we need a new batch in the dataset.py, which at the same time is called by the train.py file
