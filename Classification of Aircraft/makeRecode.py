import tensorflow as tf
import sys
import numpy as np
from PIL import Image
import cv2
import random

val_tfrecord_path='val.tfrecords'
train_tfrecord_path='train.tfrecords'
test_tfrecord_path='test.tfrecords'
testset_tfrecord_path='testset.tfrecords'
writer_val = tf.python_io.TFRecordWriter(val_tfrecord_path)
writer_train = tf.python_io.TFRecordWriter(train_tfrecord_path)
writer_test = tf.python_io.TFRecordWriter(test_tfrecord_path)
writer_testset = tf.python_io.TFRecordWriter(testset_tfrecord_path)

with open('validation_X.txt') as vd:
    val_img_path = vd.read().split('\n')
with open('train_X.txt') as td:
    train_img_path = td.read().split('\n')
with open('test_X.txt') as xd:
    test_img_path = xd.read().split('\n')
with open('testset_X.txt') as td:
    testset_img_path = td.read().split('\n')
with open('validation_Y.txt') as vdy:
    val_target_path = vdy.read().split('\n')
with open('train_Y.txt') as tdy:
    train_target_path = tdy.read().split('\n')
with open('test_Y.txt') as xdy:
    test_target_path = xdy.read().split('\n')
with open('testset_Y.txt') as tdy:
    testset_target_path = tdy.read().split('\n')

def preprocess_image(image_path):
    IMAGENET_MEAN = np.array([104., 117., 124.])
    img = cv2.imread(image_path)
    height, width, _ = img.shape
	
    if width <height:
        h = int(float(256 * height) /width)
        img = cv2.resize(img,(256, h),interpolation = cv2.INTER_AREA)
    else:
        w = int(float(256 * width) / height)
        img = cv2.resize(img,(w, 256), interpolation = cv2.INTER_AREA)
    img = img.astype(np.float32)
	
    # random 244x224 patch
    height, width, _ = img.shape
    x = random.randint(0, width - 224)
    y = random.randint(0, height - 224)
    img_cropped=img[y:y+224, x:x+ 224]
    img_cropped-=IMAGENET_MEAN
    return img_cropped

def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

for i in range(len(train_img_path)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print ('Train data: {}/{}'.format(i, len(train_img_path)))
        sys.stdout.flush()
    # Load the image
    img = preprocess_image(train_img_path[i])
    label = int(train_target_path[i])
    
    # Create a feature
    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer_train.write(example.SerializeToString())
    
writer_train.close()
sys.stdout.flush()


for i in range(len(val_img_path)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print ('Validation data: {}/{}'.format(i, len(val_img_path)))
        sys.stdout.flush()
    # Load the image
    img = preprocess_image(val_img_path[i])
    label = int(val_target_path[i])
    
    # Create a feature
    feature = {'Validation/label': _int64_feature(label),
               'Validation/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer_val.write(example.SerializeToString())
    
writer_val.close()
sys.stdout.flush()

for i in range(len(test_img_path)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print ('test data: {}/{}'.format(i, len(test_img_path)))
        sys.stdout.flush()
    # Load the image
    img = preprocess_image(test_img_path[i])
    label = int(test_target_path[i])
    
    # Create a feature
    feature = {'test/label': _int64_feature(label),
               'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer_test.write(example.SerializeToString())
    
writer_test.close()
sys.stdout.flush()

for i in range(len(testset_img_path)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print ('Testset data: {}/{}'.format(i, len(testset_img_path)))
        sys.stdout.flush()
    # Load the image
    img = preprocess_image(testset_img_path[i])
    label = int(testset_target_path[i])
    
    # Create a feature
    feature = {'testset/label': _int64_feature(label),
               'testset/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer_testset.write(example.SerializeToString())
	
writer_testset.close()
sys.stdout.flush()

print("end")

