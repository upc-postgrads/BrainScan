### GenerateTFRecords.py
Due to the nature of this dataset, where every patient is represented with 4 different volumes of the brain, the idea of creating a generator was somehow problematic. As we had to "slice" the data in 2d images we came up with the idea of slicing the first slide of every volume and storing it as if they were different channels from the same image, processing it and doing it "n" times up until the last image, and with all the patients, ending up with a really long tensor of binary data called TFRecord, compatible with the tensorflow function tf.data.TFRecordsDataset(). 

This generator starts defining some functions that allow us to:
1- Store the ID of every patient
2- Discard fully black images that would add noise to the training
3- And normalize and also store the valid images in grayscale

Having defined these helper blocks, we then execute the main function, which does the following:

1- First, it defines different "writers" with tf.python_io.TFRecordWriter for 3 different batches, which are training, validation and testing.
2- Then we use the library Nibabel which is useful for transforming .NIFTI images into raw data.
3- With this raw data in 3D we slice it in smaller 2D images and we do it with the 4 volumes from only 1 patient, so that we end up with 155 * 4 images
4- These images are parsed to become grayscale, discarted if they are fully black and stored in order in a list, which is later on converted into a nummpy array
5- We convert the labels into np.uint8
6- After that we create a dictionary of keys and values from the attributes of this patient: image, label, PatientID,...etc with the tf.train.Example function
7- Finally we convert some of the values of the dictionary from a list of bytes and int.64 to a tensor of bytes and int.64 respectively, with the tf.train.Feature()The reason is because the tf.data.TFRecordDataset() needs this structure of tensors to work within the tensorflow framework
8- Everything is done in 3 differents loops, one for every batch type (training, validation and test) that parse all the patients.
9- At the end the batch size of training and validation will depend on the % we want to keep for validating purposes
10- This generator will be called everytime we need a new batch in the dataset.py, which at the same time is called by the train.py file