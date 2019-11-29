# Welcome to Evaluate model!

Here are some instrctions for training the network.
# Requirements: software

 1. Tensorflow 1.3 or above.
 2. opencv3
# Preparation for training
update soon....
# Pre-processing

update soon....

# Files after pre-processing

After pre-processing, you will have following files.

mean_pixelrgb.npy, std_pixelrgb.npy : the mean and the standard deviation of each pixel on images in the entire dataset.

**training set:**

sf_filename.npy : A list of colorized depth image path. (shuffle)

sf_vector.npy : A list of grasp parameters. (shuffle)

sf_label.npy : A list of label. (shuffle)

**validation set**

sf_vfilename.npy : A list of colorized depth image path. (shuffle)

sf_vvector.npy : A list of grasp parameters. (shuffle)

sf_vlabel.npy : A list of label. (shuffle)

Above files are the input of the network. Please see example floder to see the right format of above files.

## Train 
It`s easy to train the network if you have above files. Just run following command:

    python train.py

It will read above files and train the network for 40 epochs. The hyperparameters are easily to change in the train.py

## Using Tensorboard to see the process

Update soon.........
