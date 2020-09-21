# Copyright (c) 2018 XLAB d.o.o
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# imports
import tensorflow as tf
import numpy as np

img_h = img_w = 28             # MNIST images are 28x28
img_size_flat = img_h * img_w + 1 # 28x28 + 1=785, the total number of pixels plus bias
n_classes = 10                 # Number of classes, one class per digit
disc_data = 10
disc_value = 1000


def load_data(mode='train'):
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: images and the corresponding labels
    """
    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    mnist = tf.keras.datasets.fashion_mnist.load_data()
    # print(mnist)
    if mode == 'train':
        x_train, y_train, x_valid, y_valid = mnist[0][0], mnist[0][1], mnist[1][0], mnist[1][1]
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        x_test, y_test = mnist[1][0], mnist[1][1]
        return x_test, y_test


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def discretize_data(d, size):
    max_value = np.amax(np.abs(d))
    res = np.floor((d / max_value) * size) / size
    return res

def add_ones(d):
    print(d.shape)
    ones = np.ones((d.shape[0], 1))
    return np.concatenate([ones, d], axis=1)


# Load MNIST data
x_train, y_train, x_valid, y_valid = load_data(mode='train')

# for i in range(len(x_valid)):
#     print("here", x_valid[i][10])

x_train = x_train.reshape(x_train.shape[0], -1)
x_valid = x_valid.reshape(x_valid.shape[0], -1)
x_train = add_ones(discretize_data(x_train, disc_data))
x_valid = add_ones(discretize_data(x_valid, disc_data))



print("Size of:")
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Validation-set:\t{}".format(len(y_valid)))


print('x_train:\t{}'.format(x_train.shape))
print('y_train:\t{}'.format(y_train.shape))
print('x_train:\t{}'.format(x_valid.shape))
print('y_valid:\t{}'.format(y_valid.shape))

inputs = tf.keras.Input(shape=(img_size_flat,))
# x1 = tf.keras.layers.Dropout(0.05)(inputs)

# x = tf.keras.layers.Dense(100, use_bias=False, kernel_initializer='glorot_normal', activation=tf.nn.relu)(inputs)
x2 = tf.keras.layers.Dense(100, use_bias=False, kernel_initializer='glorot_normal')(inputs)

# t1 = tf.keras.layers.RepeatVector(20)(x2)
# t2 = tf.keras.layers.Concatenate()([x2 for _ in range(20)])
# t3 = tf.keras.layers.Reshape((20, 20))(t2)
# t4 = tf.transpose(t3, perm=[0, 2, 1])


x3 = tf.keras.layers.Multiply()([x2, x2])
# x4 = tf.keras.layers.Flatten()(x3)


# x5 = tf.keras.layers.Dropout(0.05)(x4)

output = tf.keras.layers.Dense(10, kernel_initializer='glorot_normal', use_bias=False)(x3)

#

# inputs = tf.keras.Input(shape=(28, 28, 1))
# # x1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))(inputs)
# x1 = tf.keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3))(inputs)
# x2 = tf.keras.layers.Flatten()(x1)
# multiplied = tf.keras.layers.Multiply()([x2, x2])
#
# # output = tf.keras.layers.Dense(10, use_bias=False, kernel_initializer='glorot_normal', activation=tf.nn.softmax)(x2)
# output = tf.keras.layers.Dense(10, kernel_initializer='glorot_normal')(multiplied)



model = tf.keras.Model(inputs=inputs, outputs=output)



# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(10, input_shape=(img_size_flat, ),),
#     tf.keras.layers.Multiply()()
# ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# print(y_train.shape)

model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid))

test_loss, test_acc = model.evaluate(x_valid,  y_valid, verbose=2)

print('\nTest accuracy:', test_acc)

weights = [[], [], [], []]
i = 0
for layer in model.layers:
    # print(layer.get_weights())
    weights[i] = layer.get_weights() # list of numpy arrays
    i+=1
    # print(weights, weights[0].shape)

disc_value2 = disc_value
Pr_model = weights[1][0]
Pr = discretize_data(weights[1][0], disc_value2)
Pr = Pr * disc_value2
Pr = Pr.astype(int)

Di_model = weights[3][0]
Di = discretize_data(Di_model, disc_value)
Di = Di * disc_value
Di = Di.astype(int)

# print(Pr)

bla = []
correct = 0
correct2 = 0

for i in range(len(x_valid)):
    # true_val = model(x_valid[i])
    # index3 = np.argmax(true_val)

    v = np.matmul(x_valid[i], Pr)
    vv = np.multiply(v, v)
    vvv = np.matmul(vv, Di)
    # print(vvv)
    index = np.argmax(vvv)

    v2 = np.matmul(x_valid[i], Pr_model)
    vv2 = np.multiply(v2, v2)
    vvv2 = np.matmul(vv2, Di_model)
    index2 = np.argmax(vvv2)
    if index == y_valid[i]:
        correct += 1
    if index2 == y_valid[i]:
        correct2 += 1

    # if index2 != index3:
    #     print("strange", true_val, vvv2)
    # print(index, index2, index3, y_valid[i])

print("accuracy", float(correct2)/len(x_valid))
print("after discretization", float(correct)/len(x_valid))

# # print(Pr)
# correct = 0
# for i in range(len(x_valid)):
#     # print(x_valid[i])
#     v = Pr.transpose() * x_train[i]
#     v = v.transpose()[0]
#     index = np.argmax(v)
#     print(v)
#     print(index)
#     # print(y_valid[i])



#
#
# # Hyper-parameters
# epochs = 10             # Total number of training epochs
# batch_size = 100        # Training batch size
# display_freq = 100      # Frequency of displaying the training results
# learning_rate = 0.001   # The optimization initial learning rate
#
# h1 = 20       # The first hidden layer is a projection to h1 dimensions
#
# # weight and bais wrappers
# def weight_variable(name, shape):
#     """
#     Create a weight variable with appropriate initialization
#     :param name: weight name
#     :param shape: weight shape
#     :return: initialized weight variable
#     """
#     initer = tf.truncated_normal_initializer(stddev=0.01)
#     return tf.get_variable('W_' + name,
#                            dtype=tf.float32,
#                            shape=shape,
#                            initializer=initer)
#
#
#
# # Create the graph for the linear model
# # Placeholders for inputs (x) and outputs(y)
# x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
# y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
#
#
# P = weight_variable('projection', shape=[img_size_flat, h1])
# fc1 = tf.matmul(x, P)
#
# x2 = tf.square(fc1)
#
# D = weight_variable('output', shape=[h1, n_classes])
# output_logits = tf.matmul(x2, D)
#
# # Network predictions
# cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')
#
# # Define the loss function, optimizer, and accuracy
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
# correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
#
# # Create the op for initializing all variables
# init = tf.global_variables_initializer()
#
# sess = tf.InteractiveSession()
# sess.run(init)
# global_step = 0
# # Number of training iterations in each epoch
# num_tr_iter = int(len(y_train) / batch_size)
# for epoch in range(epochs):
#     print('Training epoch: {}'.format(epoch + 1))
#     x_train, y_train = randomize(x_train, y_train)
#     for iteration in range(num_tr_iter):
#         global_step += 1
#         start = iteration * batch_size
#         end = (iteration + 1) * batch_size
#         x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
#
#         # Run optimization op (backprop)
#         feed_dict_batch = {x: x_batch, y: y_batch}
#         sess.run(optimizer, feed_dict=feed_dict_batch)
#
#         if iteration % display_freq == 0:
#             # Calculate and display the batch loss and accuracy
#             loss_batch, acc_batch = sess.run([loss, accuracy],
#                                              feed_dict=feed_dict_batch)
#
#             print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
#                   format(iteration, loss_batch, acc_batch))
#
#     # Run validation after every epoch
#     feed_dict_valid = {x: x_valid[:5000], y: y_valid[:5000]}
#     Pr, Di, loss_valid, acc_valid = sess.run([P, D, loss, accuracy], feed_dict=feed_dict_valid)
#     print('---------------------------------------------------------')
#     print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
#           format(epoch + 1, loss_valid, acc_valid))
#     print('---------------------------------------------------------')
# # print(Di)
#
#
# Pr = discretize_data(Pr, disc_value)
# Di = discretize_data(Di, disc_value)
# predict = np.matmul(np.square(np.matmul(x_valid, Pr)), Di)
# correct = np.equal(np.argmax(predict, 1), np.argmax(y_valid, 1))
# final_acc = np.mean(correct)
# print('-----------------------------------------------------------------')
# print('-----------------------------------------------------------------')
# print('-----------------------------------------------------------------')
# print("The final accuracy of validation set after discretization: {0:.01%}".
#       format(final_acc))
# print('-----------------------------------------------------------------')
# print('-----------------------------------------------------------------')
# print('-----------------------------------------------------------------')
#
#
# def matrix_to_txt(Mat, name):
#     w = open(name + '.txt', 'w')
#     for i in range(Mat.shape[0]):
#         row = [str(x) for x in Mat[i, :]]
#         w.write(' '.join(row) + '\n')
#     w.close()
#
#
# Pr = Pr * disc_value
# Pr = Pr.astype(int)
# Di = Di * disc_value
# Di = Di.astype(int)
# valid1 = np.floor(x_valid[:1] * disc_data + 0.5)
# valid1 = valid1.astype(int)
#
# matrix_to_txt(np.transpose(Pr), 'testdata/mat_proj')
# matrix_to_txt(np.transpose(Di), 'testdata/mat_diag')
# matrix_to_txt(valid1, 'testdata/mat_valid')