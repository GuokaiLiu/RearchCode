# -*- coding: utf-8 -*-
"""
Created on Mon May  7 09:00:34 2018

@author: guokai_liu
"""

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
from os import listdir
import os
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variable_scope as vs
import matplotlib.pyplot as plt



#def image_to_tfexample(image_data, image_format, height, width, class_id):
#  return tf.train.Example(features=tf.train.Features(feature={
#      'image/encoded': bytes_feature(image_data),
#      'image/format': bytes_feature(image_format),
#      'image/class/label': int64_feature(class_id),
#      'image/height': int64_feature(height),
#      'image/width': int64_feature(width),
#  }))
#%% TFRecord Test: Making TFRecord Data File

files = tf.train.match_filenames_once(['flowers/_train*'])
filename_queue = tf.train.string_input_producer(files,shuffle=True,num_epochs=100)
reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)

features= tf.parse_single_example(
        serialized_example,
        features={
                  'image/encoded':tf.FixedLenFeature([],tf.string),
                  'image/format':tf.FixedLenFeature([],tf.string),
                  'image/class/label':tf.FixedLenFeature([],tf.int64),
                  'image/height':tf.FixedLenFeature([],tf.int64),
                  'image/width':tf.FixedLenFeature([],tf.int64),
                  'image/filename':tf.FixedLenFeature([],tf.string)
                                        })

my_height = tf.cast(features['image/height'],tf.int32)
my_width = tf.cast(features['image/width'],tf.int32)
my_id = features['image/class/label']
fn = features['image/filename']
f = features['image/format']
decoded_data = tf.decode_raw(features['image/encoded'],tf.uint8) #为何这种解码方式有问题？？？
data1 = tf.image.decode_jpeg(features['image/encoded'])
data2 = tf.reshape(data1,(my_height,my_width,3))   
data3 = tf.image.random_brightness(data2,max_delta=0.5)
data4 = tf.image.convert_image_dtype(data3,dtype=tf.float32)
#decoded_data.set_shape([my_height,my_width,3])

data5 = tf.image.resize_images(data4,[50,50],method=1) # method=2时 data4会出现负数 why？

batch_size = 5
capacity = 1000 + 3 * batch_size
capacity = 1000 + 3 * batch_size
example_batch,label_batch = tf.train.shuffle_batch([data5,my_id], 
                                        batch_size=batch_size, 
                                        capacity=capacity,
                                        min_after_dequeue=2)


c = []
d = []   
batch=True
with tf.Session() as sess:
    # Initialization
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    if batch==False:    
        # 每次运行可以读取TFRecord文件中的一个样例。当所有样例都读完之后，在此样例中程序会在重头读取
        fig = plt.figure(figsize=(8,4.5))
        for i in range(15):
            image,h,w,name,s,d2,d3,d4,d5 = sess.run([decoded_data,my_height,my_width,fn,f,data2,data3,data4,data5])
            
    #        print('File name is: %s,\nsize is w=%d x h=%d pix1=%d pix2=%d \n'%(name.decode('utf-8').split('\\')[-2:],w,h,w*h*3,len(image)))
            ax = fig.add_subplot(3,5,i+1)
    #        print('The image value is between [%f,%f]'%(d4.min(),d4.max()))
            ax.imshow(d5)
            plt.tight_layout()
    elif batch==True:
        for i in range(5):
            cur_example_batch,cur_label_batch = sess.run([example_batch,label_batch])
            c.append(cur_label_batch)
            d.append(cur_example_batch)
            print('The #%d time batch labels are:\n'%(i+1),c[i])
    coord.request_stop()
    coord.join(threads)
    
#plt.plot(c)
    

#%%
    