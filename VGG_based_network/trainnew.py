# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:29:17 2017

@author: chaoz
"""

#%%
import os
import os.path
import numpy as np
import tensorflow as tf
import VGG16N
import tools
import a0827balance2 as shuu
slim=tf.contrib.slim

#%%
#os.environ['CUDA_VISIBLE_DEVICES']='0'
N_CLASSES=2 #number of classes
MAX_STEP=100000000
num_epoch=40 #number of epochs that you want to train
    
main_dir='/bham/hpfs/gpu/projects/grasp/gcode/gm1train2019/net_rss_gm1/'

rgb_pixel=np.load(main_dir+'mean_pixelrgb.npy')
std_pixel=np.load(main_dir+'std_pixelrgb.npy')

def pre_function(filename, label, vector,four_nodes):
  filename = tf.cast(filename, tf.string)
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)

  image_standard = tf.cast(image_decoded, tf.float32)
  image_standard = (image_standard-rgb_pixel)/std_pixel
  image_standard = tf.cast(image_standard, tf.float32)
  label = tf.cast(label, tf.int32)
  label = tf.one_hot(label, depth= 2)
  vector = tf.cast(vector, tf.float32)
  four_nodes = tf.cast(four_nodes, tf.float32)
  return image_standard, label, vector, four_nodes  
#%%
def train():
    
    step=0 #step
    bs=128 #batch size
    pre_trained_weights = main_dir+'vgg16.npy' #vgg16 weight
    train_log_dir = main_dir+'trainloggm1rss/tlog' #train log path
    val_log_dir =main_dir+'trainloggm1rss/vlog'    # val log path
    train_data_dir=main_dir+'ymodellog'            # save model path
#    rd=main_dir+'modellog'
    #train data
    tra_filename=np.load(main_dir+"sf_filename.npy")
    tra_label=np.load(main_dir+"sf_label.npy")
    tra_vector=np.load(main_dir+"sf_vector.npy")
    tra_4=np.load(main_dir+"sf_4.npy")
    #val data
    val_filename=np.load(main_dir+"sf_gm1vfilename.npy")
    val_label=np.load(main_dir+"sf_gm1vlabel.npy")
    val_vector=np.load(main_dir+"sf_gm1vvector.npy")
    val_4=np.load(main_dir+"sf_gm1v4.npy")
    with tf.Graph().as_default() as g : 
        tra_image_p = tf.placeholder(tra_filename.dtype, tra_filename.shape)
        tra_label_p = tf.placeholder(tra_label.dtype, tra_label.shape)
        tra_vector_p = tf.placeholder(tra_vector.dtype, tra_vector.shape)
        tra_4_p = tf.placeholder(tra_4.dtype, tra_4.shape)
        tdataset = tf.contrib.data.Dataset.from_tensor_slices((tra_image_p, 
                                                      tra_label_p, 
                                                      tra_vector_p,tra_4_p))
        tdataset = tdataset.map(pre_function,num_threads=64)
        tdataset = tdataset.shuffle(1024*16)
        tdataset = tdataset.repeat()#重复
        tdataset = tdataset.batch(bs)
        tra_iterator = tdataset.make_initializable_iterator()
        
        val_image_p = tf.placeholder(val_filename.dtype, val_filename.shape)
        val_label_p = tf.placeholder(val_label.dtype, val_label.shape)
        val_vector_p = tf.placeholder(val_vector.dtype, val_vector.shape)
        val_4_p = tf.placeholder(val_4.dtype, val_4.shape)
        vdataset = tf.contrib.data.Dataset.from_tensor_slices((val_image_p, 
                                                      val_label_p, 
                                                      val_vector_p,val_4_p))
        vdataset = vdataset.map(pre_function)
        vdataset = vdataset.repeat()#重复
        vdataset = vdataset.batch(bs)
        val_iterator = vdataset.make_initializable_iterator() 
        # Generate placeholders for the images and labels.
        x = tf.placeholder(tf.float32, shape=[bs, 224, 224, 3])
        v = tf.placeholder(tf.float32, shape=[bs, 280])
        y_ = tf.placeholder(tf.int32, shape=[bs,2]) #??
        s_ = tf.placeholder(tf.float32, shape=[bs,4]) #??
        BN_istrain = tf.placeholder(tf.bool)
        # Build a Graph that computes predictions from the inference model.
        logits = VGG16N.VGG16N(x, N_CLASSES, v, BN_istrain)   
        # Add to the Graph the Ops for loss calculation.
        loss, mean_summary,total_loss_summary,loss_averages_op = tools.loss(logits, y_, s_)    
        # Add to the Graph the Ops that calculate and apply gradients.
        my_global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tools.optimize(loss, my_global_step,loss_averages_op)    
        # Add the Op to compare the logits to the labels during evaluation.
        accuracy,accuracy_summary = tools.accuracy(logits, y_)   
        # Build the summary Tensor based on the TF collection of Summaries.
        summary= tf.summary.merge([mean_summary, accuracy_summary,total_loss_summary])    
        # Add the variable initializer Op.
        saver = tf.train.Saver(max_to_keep=100)        
        init = tf.global_variables_initializer()    
        # Create a saver for writing training checkpoints.
        # Create a session for running Ops on the Graph.
        sess = tf.Session()
    
        # Instantiate a SummaryWriter to output summaries and the Graph.
        tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
    
        # And then after everything is built:
        # Run the Op to initialize the variables.
        sess.run(init)
        tools.load_with_skip(pre_trained_weights, sess, ['fc6','fc7','fc8'])
#        sess.run(tra_iterator.initializer, feed_dict={tra_image_p: tra_filename,tra_label_p: tra_label,tra_vector_p: tra_vector})
        sess.run(val_iterator.initializer, feed_dict={val_image_p: val_filename,val_label_p: val_label,val_vector_p: val_vector,val_4_p:val_4})
        tra_next = tra_iterator.get_next()
        val_next = val_iterator.get_next()
        print("Reading checkpoints...")

        for epoch in range(num_epoch):
          shuu.shu()
          tra_filename=np.load(main_dir+"gm1sf_filename.npy")
          tra_label=np.load(main_dir+"gm1sf_label.npy")
          tra_vector=np.load(main_dir+"gm1sf_vector.npy")
          tra_4=np.load(main_dir+"gm1sf_4.npy")     
          sess.run(tra_iterator.initializer, feed_dict={tra_image_p: tra_filename,tra_label_p: tra_label,tra_vector_p: tra_vector,tra_4_p:tra_4})         
          while True:
            try:            
                for step in range(MAX_STEP):
                    tra_all=sess.run(tra_next)
                    tra_i = tra_all[0]
                    tra_l = tra_all[1]
                    tra_v = tra_all[2]
                    tra_f = tra_all[3]                                                    
                    summary_str, _, tra_loss, tra_acc = sess.run([summary,train_op, loss, accuracy],
                                                    feed_dict={x:tra_i, y_:tra_l, v:tra_v,s_:tra_f, BN_istrain:True}) 
                         
                    if step % 20 == 0 or (step + 1) == MAX_STEP: 
                        tra_summary_writer.add_summary(summary_str, step)
#                        print ('Step: %d, loss: %.4f' % (step, tra_loss))
                        
                    if step % 20 == 0 or (step + 1) == MAX_STEP:
                        val_all=sess.run(val_next)
                        val_i = val_all[0]
                        val_l = val_all[1]
                        val_v = val_all[2]
                        val_f = val_all[3]
                        val_loss, val_acc = sess.run([loss, accuracy],
                                                     feed_dict={x:val_i, y_:val_l, v:val_v, s_:val_f, BN_istrain:False})
                        print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc))
        
                        summary_str = sess.run(summary,feed_dict={x:val_i, y_:val_l, v:val_v,s_:val_f, BN_istrain:False})
                        val_summary_writer.add_summary(summary_str, step)
                    
#                    if step == 99:  # Record execution stats
#                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#                        run_metadata = tf.RunMetadata()       
#                        summary_str, _= sess.run([summary,train_op],
#                                                    feed_dict={x:tra_i, y_:tra_l, v:tra_v, BN_istrain:True},options=run_options,run_metadata=run_metadata) 
#                        tra_summary_writer.add_run_metadata(run_metadata, 'step%d' % step)
#                        tra_summary_writer.add_summary(summary_str, step)
#                        print('Adding run metadata for', step)
                    if step % 10000 ==0:
                        checkpoint_path = os.path.join(train_data_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)               
              
            except tf.errors.OutOfRangeError:
              break
        sess.close()

if __name__ == "__main__":

	print ('working')
	train()                  
                    
