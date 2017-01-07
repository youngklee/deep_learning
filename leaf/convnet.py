import csv
import random
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

IMAGE_SIZE = 224
# After two 2x2 pooling, the image size should a quater of the original.
POOLED_IMAGE_SIZE = int(IMAGE_SIZE/4)

def read_classes():
    with open('classes.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        classes = next(reader)
        return classes

classes = read_classes()
print '%i classes' % len(classes)

def encode_label(label):
    new_label = len(classes)*[0]
    new_label[classes.index(label)] = 1
    return new_label

def decode_label(label):
    return classes[label]

def labeled_image_filenames(csv_filename):
    prefix = './resized_images/'
    postfix = '.jpg'
    image_filenames = []
    labels = []
    with open(csv_filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # Skip the headers
        for row in reader:
            image_filenames.append(prefix + row[0] + postfix)
            labels.append(encode_label(row[1]))
    return image_filenames, labels

def read_image(queue):
    label = queue[1]
    image_file = tf.read_file(queue[0])
    image = tf.cast(tf.image.decode_jpeg(image_file), tf.float32)
    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 1))
    return image, label

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')

def weight_variable(shape, name = ''):
    init = tf.truncated_normal(shape, stddev=0.1)
    if name:
        return tf.Variable(init, name = name)
    else:
        return tf.Variable(init)

def bias_variable(shape, name = ''):
    init = tf.constant(0.1, shape=shape)
    if name:
        return tf.Variable(init, name = name)
    else:
        return tf.Variable(init)

def model(x, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2, keep_prob):
    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, POOLED_IMAGE_SIZE*POOLED_IMAGE_SIZE*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y

W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
b_conv1 = bias_variable([32], 'b_conv1')
W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
b_conv2 = bias_variable([64], 'b_conv2')
W_fc1 = weight_variable([POOLED_IMAGE_SIZE*POOLED_IMAGE_SIZE*64, 1024], 'W_fc1')
b_fc1 = bias_variable([1024], 'b_fc1')
W_fc2 = weight_variable([1024, 100], 'W_fc2')
b_fc2 = bias_variable([100], 'b_fc2')

training = input('Would you like to train a model: ')
training = int(training)
if training:
    iteration = input('Enter no. of iterations: ')
    iteration = int(iteration)
    image_filenames, labels = labeled_image_filenames('train.csv')
    print '%i images' % len(image_filenames)

    test_set_size = int(0.2*len(image_filenames) + 0.5)
    print '%i training images' % (len(image_filenames) - test_set_size)
    print '%i test images' % test_set_size

    images = ops.convert_to_tensor(image_filenames, dtype = dtypes.string)
    labels = ops.convert_to_tensor(labels, dtype = dtypes.float32)

    partitions = [0]*len(image_filenames)
    partitions[:test_set_size] = [1]*test_set_size
    random.shuffle(partitions)

    train_images, test_images = tf.dynamic_partition(images, partitions, 2)
    train_labels, test_labels= tf.dynamic_partition(labels, partitions, 2)

    train_image_queue = tf.train.slice_input_producer(
            [train_images, train_labels],
            shuffle = False)

    test_image_queue = tf.train.slice_input_producer(
            [test_images, test_labels],
            shuffle = False)

    train_image, train_label = read_image(train_image_queue)
    train_batch = tf.train.batch(
            [train_image, train_label],
            batch_size = 50)

    test_image, test_label = read_image(test_image_queue)
    test_batch = tf.train.batch(
            [test_image, test_label],
            batch_size = 50)

    keep_prob = tf.placeholder(tf.float32)
    x = train_batch[0]
    y_ = train_batch[1]
    y = model(x, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2, keep_prob)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord = coord)

    print 'Start training'

    for i in range(iteration):
        if i % 100 == 0:
            print(sess.run(accuracy, feed_dict = {keep_prob: 1.0}))
        sess.run(train_step, feed_dict = {keep_prob: 0.5})

    saver = tf.train.Saver()
    saver.save(sess, './leaf.ckpt')

    x = test_batch[0]
    y_ = test_batch[1]

    print("test accuracy %g"%sess.run(accuracy, feed_dict = {keep_prob: 1.0}))

    coord.request_stop()
    coord.join(threads)

else:
    sess = tf.Session()
    #sess.run(tf.global_variables_initializer())
    #saver = tf.train.import_meta_graph('leaf.meta')
    #saver.restore(sess, tf.train.latest_checkpoint('./'))
    saver = tf.train.Saver()
    saver.restore(sess, './leaf.ckpt')

    x = tf.placeholder(tf.float32, [None, 784])
    keep_prob = tf.placeholder(tf.float32)
    y = model(x, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2, keep_prob)
    prediction = tf.argmax(y, 1)

    def get_label(filepath):
        image_filenames, labels = labeled_image_filenames('train.csv')
        index = image_filenames.index(filepath)
        return decode_label(labels[index].index(1))

    id_ = 1
    while id_:
        id_ = input('Enter id: ')
        if id_:
            filepath = './resized_images/' + str(id_) + '.jpg'
            image_file  = tf.read_file(filepath)
            image = tf.image.decode_jpeg(image_file)
            image = tf.cast(image, tf.float32)
            image = tf.reshape(image, [1, IMAGE_SIZE*IMAGE_SIZE]).eval(session = sess)
            try:
                label = get_label(filepath)
                print(decode_label(sess.run(prediction, feed_dict = {x: image, keep_prob: 1.0})[0]), label)
            except ValueError:
                print('The id does not exist.')


