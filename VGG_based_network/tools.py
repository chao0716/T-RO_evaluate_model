# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:09:38 2017

@author: chaoz
"""
import tensorflow as tf
import numpy as np

#%%
def _variable_on_cpu(name, trainable, shape, initializer):
  """
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = tf.get_variable(name, shape, trainable=trainable, initializer=initializer, dtype=dtype)
  return var
#%%
def _variable_with_weight_decay(name, trainable, shape, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(
      name,
      trainable,
      shape,
      tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay*0.01)
  return var
#%%
def conv(layer_name, x, out_channels, kernel_size, stride,istrainable=True):
    '''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        trainable:
    Returns:
        4D tensor
    '''

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = _variable_with_weight_decay(name='weights',
                            trainable=istrainable,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            wd=0.004) # default is uniform distribution initialization
        b = _variable_on_cpu(name='biases',
                            trainable=istrainable,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.1))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x
#%%
def pool(layer_name, x, kernel, stride, is_max_pool):
    '''
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding:
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    '''
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='VALID',name=layer_name)
    return x
#%%
def FC_layer(layer_name, x, out_nodes,istrainable=True):
    '''
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
    '''
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = _variable_with_weight_decay('weights',
                            trainable=istrainable,
                            shape=[size, out_nodes],
                            wd=0.004)
        b = _variable_on_cpu('biases',
                             trainable=istrainable,
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.1))
        flat_x = tf.reshape(x, [-1, size]) # flatten into 1D
        
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
    return x
#%%
def Softmax_layer(layer_name, x, out_nodes,istrainable=True):
    '''
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
    '''
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = _variable_with_weight_decay('weights',
                            trainable=istrainable,
                            shape=[size, out_nodes],
                            wd=None)
        b = _variable_on_cpu('biases',
                             trainable=istrainable,
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        
        x = tf.nn.bias_add(tf.matmul(x, w), b)
    return x
#%%
def loss(logits, labels, f_4_l):
    '''Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    '''
    with tf.name_scope('loss') as scope:       
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        
        tf.add_to_collection('losses', cross_entropy_mean)
        
        total_loss=tf.add_n(tf.get_collection('losses'), name='total_loss')
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        mean_summary=tf.summary.scalar(scope+'/cross_entropy_mean1', cross_entropy_mean)
        total_loss_summary=tf.summary.scalar(scope+'/total_loss1', total_loss)
             
    return total_loss, mean_summary, total_loss_summary,loss_averages_op
#%%
def accuracy(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, 
  """
  with tf.name_scope('accuracy') as scope:
      correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
      correct = tf.cast(correct, tf.float32)
      accuracy = tf.reduce_mean(correct)*100.0
      accuracy_summary=tf.summary.scalar(scope+'/accuracy', accuracy)
  return accuracy,accuracy_summary
#%%
def optimize(loss, global_step,loss_averages_op):
    '''optimization, use Gradient Descent as default
    '''
    starter_learning_rate = 0.01
    learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,80000, 0.5, staircase=True)
#    learning_rate=0.005
#    learning_rate=0.001
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
#        opt = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=0.1)
        grads = opt.compute_gradients(loss)
    
      # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        
      # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
        curr_lr = opt._learning_rate
        curr_lr_summary=tf.summary.scalar('curr_lr', curr_lr)           
        return train_op
#%%
def num_correct_prediction(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Return:
      the number of correct predictions
  """
  correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
  correct = tf.cast(correct, tf.int32)
  n_correct = tf.reduce_sum(correct)
  return n_correct       
#%%
def load(data_path, session):
    data_dict = np.load(data_path, encoding='latin1').item()
    
    keys = sorted(data_dict.keys())
    for key in keys:
        with tf.variable_scope(key, reuse=True):
            for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                session.run(tf.get_variable(subkey).assign(data))                  
#%%                
def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding='latin1').item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))   
