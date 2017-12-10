import tensorflow as tf
import numpy as np
import time
import math
import random

class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.last_layer_parameters = []     ## Parameters in this list will be optimized when only last layer is being trained 
        self.parameters = []                ## Parameters in this list will be optimized when whole BCNN network is finetuned
        self.convlayers()                   ## Create Convolutional layers
        self.fc_layers()                    ## Create Fully connected layer
        self.weight_file = weights

    def convlayers(self):
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean
            print('Adding Data Augmentation')
            
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                   trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64],  dtype=tf.float32),
                                   trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                   trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1),  trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                   trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                   trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1),  trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                   trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,   name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1),  trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        print('Shape of conv5_3', self.conv5_3.get_shape())
        self.phi_I = tf.einsum('ijkm,ijkn->imn',self.conv5_3,self.conv5_3)
        print('Shape of phi_I after einsum', self.phi_I.get_shape())

        self.phi_I = tf.reshape(self.phi_I,[-1,512*512])
        print('Shape of phi_I after reshape', self.phi_I.get_shape())

        self.phi_I = tf.divide(self.phi_I,784.0)  
        print('Shape of phi_I after division', self.phi_I.get_shape())

        self.y_ssqrt = tf.multiply(tf.sign(self.phi_I),tf.sqrt(tf.abs(self.phi_I)+1e-12))
        print('Shape of y_ssqrt', self.y_ssqrt.get_shape())

        self.z_l2 = tf.nn.l2_normalize(self.y_ssqrt, dim=1)
        print('Shape of z_l2', self.z_l2.get_shape())

    def fc_layers(self):

        with tf.name_scope('fc-new') as scope:
            fc3w = tf.get_variable('weights', [512*512, 100], trainable=True)
            fc3b = tf.Variable(tf.constant(1.0, shape=[100], dtype=tf.float32), name='biases', trainable=True)
            self.fc3l = tf.nn.bias_add(tf.matmul(self.z_l2, fc3w), fc3b)
            self.last_layer_parameters += [fc3w, fc3b]
            self.parameters += [fc3w, fc3b]

    def load_weights(self, sess):
        weights = np.load(self.weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            removed_layer_variables = ['fc6_W','fc6_b','fc7_W','fc7_b','fc8_W','fc8_b']
            if not k in removed_layer_variables:
                print(k)
                print("",i, k, np.shape(weights[k]))
                sess.run(self.parameters[i].assign(weights[k]))

val_tfrecord_path='val.tfrecords'
train_tfrecord_path='train.tfrecords'
test_tfrecord_path='test.tfrecords'
testset_tfrecord_path='testset.tfrecords'

sess = tf.Session()
with tf.device('/gpu:0'):
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    target = tf.placeholder("float", [None, 100])
    print ('Creating graph')
    vgg = vgg16(imgs, 'epoch_100_0.npz', sess)

training_epochs = 20
learning_rate = 0.0001
batch_size = 20

# Defining other ops using Tensorflow
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc3l, labels=target))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
cost_summ = tf.summary.scalar("cost", cost)

correct_prediction = tf.equal(tf.argmax(vgg.fc3l,1), tf.argmax(target,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)
num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

# IF WANT RECALL
# output = tf.argmax(vgg.fc3l, 1)
# x=np.empty(0, dtype='int64')
# y=np.empty(0, dtype='int64')
# x = tf.placeholder(tf.int64)
# y = tf.placeholder(tf.int64)
# acc, acc_op = tf.metrics.accuracy(labels=x, predictions=y)
# rec, rec_op = tf.metrics.recall(labels=x, predictions=y)
# pre, pre_op = tf.metrics.precision(labels=x, predictions=y)

def read_train_tfrecode():
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([train_tfrecord_path],
                                                    num_epochs=training_epochs)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32)

    # Cast label data into int32
    label = tf.cast(features['train/label'], tf.int32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [224, 224, 3])

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, 
                                            capacity=5*batch_size, num_threads=4, 
                                            min_after_dequeue=batch_size)
    labels = tf.one_hot(labels, depth=100)
    labels = tf.reshape(labels, [batch_size, 100])

    image_batch = tf.placeholder_with_default(images, shape=[batch_size, 224, 224, 3])
    label_batch = tf.placeholder_with_default(labels, shape=[batch_size, 100])
    return image_batch, label_batch

def read_val_tfrecode():
    feature = {'Validation/image': tf.FixedLenFeature([], tf.string),
               'Validation/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([val_tfrecord_path], 
                                                    num_epochs=training_epochs)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['Validation/image'], tf.float32)

    # Cast label data into int32
    label = tf.cast(features['Validation/label'], tf.int32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [224, 224, 3])

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, 
                                            capacity=batch_size*5, num_threads=4, 
                                            min_after_dequeue=batch_size)
    labels = tf.one_hot(labels, depth=100)
    labels = tf.reshape(labels, [batch_size, 100])

    image_batch = tf.placeholder_with_default(images, shape=[batch_size, 224, 224, 3])
    label_batch = tf.placeholder_with_default(labels, shape=[batch_size, 100])
    return image_batch, label_batch

def read_test_tfrecode():
    feature = {'test/image': tf.FixedLenFeature([], tf.string),
               'test/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([test_tfrecord_path], 
                                                    num_epochs=2)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['test/image'], tf.float32)

    # Cast label data into int32
    label = tf.cast(features['test/label'], tf.int32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [224, 224, 3])

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, 
                                            capacity=batch_size*5, num_threads=4, 
                                            min_after_dequeue=batch_size)
    labels = tf.one_hot(labels, depth=100)
    labels = tf.reshape(labels, [batch_size, 100])

    image_batch = tf.placeholder_with_default(images, shape=[batch_size, 224, 224, 3])
    label_batch = tf.placeholder_with_default(labels, shape=[batch_size, 100])
    return image_batch, label_batch

def read_testset_tfrecode():
    feature = {'testset/image': tf.FixedLenFeature([], tf.string),
               'testset/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([testset_tfrecord_path], 
                                                    num_epochs=2)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['testset/image'], tf.float32)

    # Cast label data into int32
    label = tf.cast(features['testset/label'], tf.int32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [224, 224, 3])

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=1, 
                                            capacity=5, num_threads=4, 
                                            min_after_dequeue=1)
    labels = tf.one_hot(labels, depth=100)
    labels = tf.reshape(labels, [1, 100])

    image_batch = tf.placeholder_with_default(images, shape=[1, 224, 224, 3])
    label_batch = tf.placeholder_with_default(labels, shape=[1, 100])
    return image_batch, label_batch

train_image_batch, train_label_batch = read_train_tfrecode()
val_image_batch, val_label_batch = read_val_tfrecode()
test_image_batch, test_label_batch = read_test_tfrecode()
testset_image_batch, testset_label_batch = read_testset_tfrecode()
summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/vgg_r0_01")
writer.add_graph(sess.graph)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
vgg.load_weights(sess)
saver = tf.train.Saver()

with tf.device('/gpu:0'):
    global_step=1
    total_batch = int(6000/batch_size)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for epoch in range(training_epochs):
        for step in range(total_batch):
            start = time.time()
            train_X, train_Y = sess.run([train_image_batch, train_label_batch])
            s, cost_value, _ = sess.run([summary, cost, optimizer], feed_dict={imgs: train_X, target: train_Y})
            writer.add_summary(s, global_step=global_step)
            global_step += 1
            if step % 20 == 0:
                print('Last layer training, time to run optimizer for batch size:', batch_size,'is --> ', round(time.time()-start,3),'seconds')
            if step % 100 == 0:
                print("Epoch:", '%03d' % (epoch+1), "Step:", '%03d' % step,"Cost:", str(cost_value))
                print("Training Accuracy -->", sess.run(accuracy,feed_dict={imgs: train_X, target: train_Y}))

        total_val_batch = int(666/batch_size)
        correct_val_count = 0
        val_cost = 0.0

        for i in range(total_val_batch):
            val_X, val_Y = sess.run([val_image_batch, val_label_batch])
            val_cost += sess.run(cost, feed_dict={imgs: val_X, target: val_Y})
            pred = sess.run(num_correct_preds, feed_dict = {imgs: val_X, target: val_Y})
            correct_val_count+=pred

        print("##############################")
        print("Epoch : %03d" % (epoch+1))
        print("Validation Cost -->", val_cost)
        print("correct_val_count, total_val_count", correct_val_count, 666)
        print("Validation Data Accuracy -->", 100.0*correct_val_count/(1.0*666))
        print("##############################")
        
    total_test_count = 3333
    correct_test_count = 0
    test_batch_size = 20
    total_test_batch = int(total_test_count/test_batch_size)
    for i in range(total_test_batch):
        test_X, test_Y = sess.run([test_image_batch, test_label_batch])
        pred = sess.run(num_correct_preds, feed_dict = {imgs: test_X, target: test_Y})
        correct_test_count+=pred

    print("##############################")
    print("correct_test_count, total_test_count", correct_test_count, total_test_count)
    print("Test Data Accuracy -->", 100.0*correct_test_count/(1.0*total_test_count))
    print("##############################")
    
    total_testset_count = 534
    correct_testset_count = 0
    for i in range(total_testset_count):
        testset_X, testset_Y = sess.run([testset_image_batch, testset_label_batch])
        pred = sess.run(num_correct_preds, feed_dict = {imgs: testset_X, target: testset_Y})
        correct_testset_count+=pred
        
    print("##############################")
    print("correct_testset_count, total_testset_count", correct_testset_count, total_testset_count)
    print("Testset Data Accuracy -->", 100.0*correct_testset_count/(1.0*total_testset_count))
    print("##############################")
    
    coord.request_stop()
    coord.join(threads)


# In[ ]:


np.savez('epoch_100_5.npz',
         conv1_1_W=sess.run(vgg.parameters[0]),
         conv1_1_b=sess.run(vgg.parameters[1]),
         conv1_2_W=sess.run(vgg.parameters[2]),
         conv1_2_b=sess.run(vgg.parameters[3]),
         conv2_1_W=sess.run(vgg.parameters[4]),
         conv2_1_b=sess.run(vgg.parameters[5]),
         conv2_2_W=sess.run(vgg.parameters[6]),
         conv2_2_b=sess.run(vgg.parameters[7]),
         conv3_1_W=sess.run(vgg.parameters[8]),
         conv3_1_b=sess.run(vgg.parameters[9]),
         conv3_2_W=sess.run(vgg.parameters[10]),
         conv3_2_b=sess.run(vgg.parameters[11]),
         conv3_3_W=sess.run(vgg.parameters[12]),
         conv3_3_b=sess.run(vgg.parameters[13]),
         conv4_1_W=sess.run(vgg.parameters[14]),
         conv4_1_b=sess.run(vgg.parameters[15]),
         conv4_2_W=sess.run(vgg.parameters[16]),
         conv4_2_b=sess.run(vgg.parameters[17]),
         conv4_3_W=sess.run(vgg.parameters[18]),
         conv4_3_b=sess.run(vgg.parameters[19]),
         conv5_1_W=sess.run(vgg.parameters[20]),
         conv5_1_b=sess.run(vgg.parameters[21]),
         conv5_2_W=sess.run(vgg.parameters[22]),
         conv5_2_b=sess.run(vgg.parameters[23]),
         conv5_3_W=sess.run(vgg.parameters[24]),
         conv5_3_b=sess.run(vgg.parameters[25]),
         weights=sess.run(vgg.parameters[26]))
print("Last layer weights saved")

# saver.save(sess, 'model',global_step=global_step)
sess.close()


# IF WANT RECALL
# with tf.device('/gpu:0'):
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#     total_testset_count = 534
# #     correct_testset_count = 0
# #     testset_batch_size = batch_size
# #     total_testset_batch = int(total_testset_count/testset_batch_size)
#     for i in range(total_testset_count):
#         testset_X, testset_Y = sess.run([testset_image_batch, testset_label_batch])
#         goal = tf.argmax(testset_Y, 1)
#         a=sess.run(goal, feed_dict={target: testset_Y})
#         b=sess.run(output, feed_dict={imgs: testset_X}) # predictions
#         v = sess.run(acc_op, feed_dict={x: a ,y: b}) #accuracy
#         r = sess.run(rec_op, feed_dict={x: a, y: b}) #recall
#         p = sess.run(pre_op, feed_dict={x: a, y: b}) #precision
        
#     print("accuracy %f", v)
#     print("recall %f", r)
#     print("precision %f", p)

# #     print("##############################")
# #     print("correct_testset_count, total_test_count", correct_testset_count, total_testset_count)
# #     print("Testset Data Accuracy -->", 100.0*correct_testset_count/(1.0*total_testset_count))
# #     print("##############################")
#     coord.request_stop()
#     coord.join(threads)

