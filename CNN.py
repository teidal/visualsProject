import numpy as np
import Camera as c
import letterRecog as lr
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import collections
import re, os, cv2
from glob import glob

#create database
def load_base(filename):
    with np.load(filename) as data:
        pictures = data["pictures"]
        labels = data ["labels"]
        return pictures, labels

def cnn():
    pictures, labels = load_base('base.npz')
    picture_test = "pictures\phone.jpg"  # for testing

    #parameters used in the graph
    learning_rate = 0.0001 #learning rate
    epochs = 10 # epochs
    batch_size = 50 #number of samples that going to be propagated through the network
    batch_iteration = 1000 #number of passes, each pass using batch_size number of examples.
    num_classes = 62 # Number of classes, one class for each of 10 digits.

    #parameters used in the cnn
    img_height = 56
    img_width = 56
    img_size_flat = img_height * img_width # Images are stored in one-dimensional arrays of this length.
    img_shape = (img_height, img_width) # Tuple with height and width of images used to reshape arrays.
    #placeholders for CNN
    x = tf.placeholder(tf.float32, [None, img_size_flat])
    y = tf.placeholder(tf.float32, [None, num_classes])
    x_shaped = tf.reshape(x, [-1, img_height, img_width, 1])
    #creating own dataset
    assert pictures.shape[0] == labels.shape[0]
    pictures_placeholder = tf.placeholder(pictures.dtype, pictures.shape)
    labels_placeholder= tf.placeholder(labels.dtype, labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices((pictures_placeholder, labels_placeholder))
    iterator = dataset.make_initializable_iterator()

#new CNN layer
#create 2 layers
    layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
    layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
    flattened = tf.reshape(layer2, [-1, 14 * 14 * 64])

# setup some weights and bias values for this layer, then activate with ReLU
    wd1 = tf.Variable(tf.truncated_normal([14 * 14 * 64, 1000], stddev=0.03), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
    dense_layer1 = tf.matmul(flattened, wd1) + bd1
    dense_layer1 = tf.nn.relu(dense_layer1)
    # another layer with softmax activations
    wd2 = tf.Variable(tf.truncated_normal([1000, num_classes], stddev=0.03), name='wd2')
    bd2 = tf.Variable(tf.truncated_normal([num_classes], stddev=0.01), name='bd2')
    dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
    y_ = tf.nn.softmax(dense_layer2)

    #cost fuction
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))
    #optimazer
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #initiator
    init_op = tf.global_variables_initializer()

    # setup recording variables
    # add a summary to store the accuracy
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('results')
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op, feed_dict={pictures_placeholder: pictures, labels_placeholder:labels})
        total_batch = int(batch_iteration / batch_size)
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(total_batch):
                #batch_x, batch_y=
                _, c = sess.run([optimiser, cost], feed_dict={x: pictures_placeholder, y: labels_placeholder})
                epoch_loss += c / total_batch
            test_acc = sess.run(accuracy, feed_dict={x: picture_test, y: labels_placeholder})
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(epoch_loss), " test accuracy: {:.3f}".format(test_acc))
            summary = sess.run(merged, feed_dict={x: pictures_placeholder, y: labels_placeholder})
            writer.add_summary(summary, epoch)
        print("\nTraining complete!")
        writer.add_graph(sess.graph)
        print(sess.run(accuracy, feed_dict={x: picture_test, y: labels_placeholder}))

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters] #setup the filter input shape for tf.nn.conv_2d
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W') #initialise weights and bias for the filter
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME') # setup the convolutional layer operation

    out_layer += bias # add the bias
    out_layer = tf.nn.relu(out_layer) # apply a ReLU non-linear activation

    ksize = [1, pool_shape[0], pool_shape[1], 1] # now perform max pooling
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME') #operation of creation the CNN layer
    return out_layer

if __name__ == "__main__":
    cnn()












