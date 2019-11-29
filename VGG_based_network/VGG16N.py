# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:05:37 2018

@author: chaozh99
"""

import tensorflow as tf
import tools
#%%
def VGG16N(x, n_classes, v, BN_istrain=True,istrainable=True):
    I_trainable=True
#    H_trainable=True    
    with tf.name_scope('VGG_conv_layers'):
        x = tools.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1],istrainable=I_trainable)
        x = tools.conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1],istrainable=I_trainable)
        x = tools.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
        #    
        x = tools.conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1], istrainable=I_trainable)   
        x = tools.conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1], istrainable=I_trainable)
        x = tools.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
        #    
        x = tools.conv('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1], istrainable=I_trainable)   
        x = tools.conv('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1], istrainable=I_trainable)    
        x = tools.conv('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1], istrainable=I_trainable)
        x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
        #    
        x = tools.conv('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], istrainable=I_trainable)   
        x = tools.conv('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], istrainable=I_trainable)
        x = tools.conv('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], istrainable=I_trainable) 
        x = tools.pool('pool4', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
        
        x = tools.conv('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], istrainable=I_trainable)       
        x = tools.conv('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], istrainable=I_trainable)          
        x = tools.conv('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], istrainable=I_trainable)      
        x = tools.FC_layer('fcc', x, out_nodes=1024, istrainable=istrainable)
        
    with tf.name_scope('Hand_conv_layers'):
        v = tools.FC_layer('fcv', v, out_nodes=1024, istrainable=istrainable)

        
    mid=tf.add(x,v)       
    mid=tools.FC_layer('fc6', mid, out_nodes=1024, istrainable=istrainable)
    mid= tf.layers.dropout(inputs=mid,rate=0.5,training=BN_istrain)
    
    mid=tools.FC_layer('fc7', mid, out_nodes=1024, istrainable=istrainable)       
    mid= tf.layers.dropout(inputs=mid,rate=0.5,training=BN_istrain)

    mid=tools.FC_layer('fc8', mid, out_nodes=1024, istrainable=istrainable)
    mid= tf.layers.dropout(inputs=mid,rate=0.5,training=BN_istrain)
    
    mid=tools.FC_layer('fc9', mid, out_nodes=1024, istrainable=istrainable)       
    mid= tf.layers.dropout(inputs=mid,rate=0.5,training=BN_istrain)
    
    mid = tools.Softmax_layer('fc11', mid, out_nodes=2,istrainable=istrainable)
    return mid

