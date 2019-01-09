from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
from pandas import get_dummies
import matplotlib.colors as cols
import math

datafile = 'fingerprint.npy'
labelfile = 'labels.npy'
data = np.load(datafile)
shape = data.shape

img_size = shape[1]
img_shape = [shape[1], shape[2]]
img_size_flat = shape[1] * shape[2]
num_channels = 1

data = data.reshape((shape[0], shape[1] * shape[2]))
data = np.asarray([d/np.max(d) for d in data])

labels_dummy = np.load(labelfile)
N = len(data)
num_classes = 2

labels = get_dummies(labels_dummy).values


batch_size = 100

num_filters1 = 32
filter_size1 = 5
num_filters2 = 64
filter_size2 = 5
fc_size = 128


def plot_image(image):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image.reshape(img_shape), norm=cols.LogNorm())
    ax.set_yticks([])
    ax.set_xticks([])


def random_batch(X_data, y_data):

    ind = np.random.randint(len(X_data), size=batch_size)
    X_batch = X_data[ind]
    y_batch = y_data[ind]

    return X_batch, y_batch


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)


layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)


layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv2)
layer_fc1 = new_fc_layer(input=layer_flat,
                    num_inputs=num_features,
                    num_outputs=fc_size,
                    use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                     num_inputs=fc_size,
                     num_outputs=num_classes,
                     use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


data_tf = tf.constant(data.astype(np.float32))
labels_tf = tf.constant(labels.astype(np.int32))
learning_rate = 0.01


session = tf.Session()
session.run(tf.global_variables_initializer())


tf.summary.scalar('Validation_Accuracy', accuracy)
merge = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('./logs/train', graph=session.graph_def)
total_iterations = 0


def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()
    acc_mat = []
    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch(data, labels)


        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # feed_dict_train = {x: data,
        #                   y_true: labels}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        #session.run(optimizer, feed_dict=feed_dict_train)
        _, acc, summary = session.run([optimizer, accuracy, merge], feed_dict=feed_dict_train)
        train_writer.add_summary(summary=summary, global_step=i)

        acc_mat.append(acc)
        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            #acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    return acc_mat

def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    label_dict = {0:'Perovskite', 1:'Non-Perovskite'}
    label_names = ['Perovskite', 'Non-Perovskite']
    cls_true = labels_dummy

    cls_true = [label_dict[c] for c in cls_true]
    cls_pred = [label_dict[c] for c in cls_pred]

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred, labels=label_names)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)

    # Make various adjustments to the plot.
    #plt.colorbar()
    ax.set_title('Confusion matrix of the classifier', fontsize=20)
    plt.colorbar(cax)
    ax.set_xticklabels([''] + label_names)
    ax.set_yticklabels([''] + label_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    for i in range(2):
        for j in range(2):
            ax.text(i,j,cm[i, j], fontdict={'fontsize':20, 'color':'white'})

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_conv_weights(weights, input_channel=0, grid=None):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_filters)))

    # Create figure with a grid of sub-plots.
    if grid is None:
        fig, axes = plt.subplots(num_grids, num_grids)
    else:
        fig, axes = plt.subplots(grid[0], grid[1])

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    #plt.show()


def plot_conv_layer(layer, image, grid=None):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_filters)))

    # Create figure with a grid of sub-plots.
    if grid is None:
        fig, axes = plt.subplots(num_grids, num_grids)
    else:
        fig, axes = plt.subplots(grid[0], grid[1])

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


acc_mat = optimize(num_iterations=700)
acc_mat = 100 * np.asarray(acc_mat)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(acc_mat, linewidth=2)
ax.set_xlabel('Iterations')
ax.set_ylabel('Accuracy')
ax.xaxis.label.set_size(26)
ax.yaxis.label.set_size(26)
for xtick, ytick in zip(ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()):
    xtick.label.set_fontsize(20)
    ytick.label.set_fontsize(20)
plt.savefig('figs/accuracy.png', bbox_inches='tight', dpi=300)


image = data[10]

plot_image(image)
plot_conv_weights(weights_conv1)
plot_conv_layer(layer=layer_conv1, image=data[10])
plot_conv_weights(weights_conv2, input_channel=10)
plot_conv_layer(layer=layer_conv2, image=data[10])

