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
disc_data = 4
disc_value = 200
hid_dim = 40


def load_data(mode='train'):
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: images and the corresponding labels
    """
    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    mnist = tf.keras.datasets.mnist.load_data()
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
inputs2 = tf.keras.layers.Dropout(0.1)(inputs)

# x = tf.keras.layers.Dense(20, kernel_initializer='glorot_normal', activation=tf.nn.relu)(inputs)
x = tf.keras.layers.Dense(hid_dim, use_bias=False, kernel_initializer='glorot_normal')(inputs2)
multiplied = tf.keras.layers.Multiply()([x, x])
multiplied2 = tf.keras.layers.Dropout(0.1)(multiplied)

output = tf.keras.layers.Dense(10, use_bias=False)(multiplied2)

model = tf.keras.Model(inputs=inputs, outputs=output)



# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(10, input_shape=(img_size_flat, ),),
#     tf.keras.layers.Multiply()()
# ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# print(y_train.shape)
check_point = tf.keras.callbacks.ModelCheckpoint('testdata/mnist_best_model.h5', monitor='val_loss', save_best_only=True)
model.fit(x_train, y_train, epochs=15, validation_data=(x_valid, y_valid), callbacks=[check_point])
model = tf.keras.models.load_model('testdata/mnist_best_model.h5')

# test_loss, test_acc = model.evaluate(x_valid,  y_valid, verbose=2)
#
# print('\nTest accuracy:', test_acc)

weights = [[], [], [], [],[],[]]
i = 0
for layer in model.layers:
    # print(layer.get_weights())
    weights[i] = layer.get_weights() # list of numpy arrays
    i+=1
    # print(weights, weights[0].shape)

disc_value2 = disc_value
disc_value2 = 200
Pr_model = weights[2][0]
Pr = discretize_data(weights[2][0], disc_value2)
Pr = Pr * disc_value2
Pr = Pr.astype(int)

Di_model = weights[5][0]
disc_value3 = 40

Di = discretize_data(Di_model, disc_value)
Di = Di * disc_value
Di = Di.astype(int)
# Di = Di + disc_value

Di3 = discretize_data(Di_model, disc_value3)
Di3 = Di3 * disc_value3
Di3 = Di3.astype(int)


x_valid2 = np.floor(x_valid * disc_data + 0.5)
x_valid2 = x_valid2.astype(int)

# print(Pr)

bla = []
correct = 0
correct2 = 0
correct3 = 0
max_bits = 0
max_bits2 = 0

for i in range(len(x_valid)):
    # true_val = model(x_valid[i])
    # index3 = np.argmax(true_val)
    # print(x_valid2[i])
    v = np.matmul(x_valid2[i], Pr)
    vv = np.multiply(v, v)
    vvv = np.matmul(vv, Di)

    bits = [np.ceil(np.log2(np.abs(x))) for x in vvv]
    # print(bits)
    max_bits = max(max_bits, max(bits))
    index = np.argmax(vvv)

    vvv3 = np.matmul(vv, Di3)
    index3 = np.argmax(vvv3)
    bits2 = [np.ceil(np.log2(np.abs(x))) for x in vvv3]
    # print(bits)
    max_bits2 = max(max_bits2, max(bits2))


    v2 = np.matmul(x_valid[i], Pr_model)
    vv2 = np.multiply(v2, v2)
    vvv2 = np.matmul(vv2, Di_model)
    index2 = np.argmax(vvv2)
    if index == y_valid[i]:
        correct += 1
    if index2 == y_valid[i]:
        correct2 += 1
    if index3 == y_valid[i]:
        correct3 += 1
    # if index2 != index3:
    #     print("strange", true_val, vvv2)
    # print(index, index2, index3, y_valid[i])

print("accuracy", float(correct2)/len(x_valid))
print("after discretization", float(correct)/len(x_valid))
print("after discretization3", float(correct3)/len(x_valid))
print("max_bits", max_bits)
print("max_bits2", max_bits2)

def matrix_to_txt(Mat, name):
    w = open(name + '.txt', 'w')
    for i in range(Mat.shape[0]):
        row = [str(x) for x in Mat[i, :]]
        w.write(' '.join(row) + '\n')
    w.close()

for i in range(10):
    d = np.diag(Di3[:,i])
    # d = np.sqrt(d)
    m = np.matmul(np.matmul(Pr, d), np.transpose(Pr))
    norm = np.linalg.norm(m, ord=2)
    bound = norm *16*785
    print("norm", norm, bound, np.log2(bound))

# norm_pr = np.linalg.norm(Pr, ord=2)
# print("norm", norm_pr, np.linalg.norm(Pr, ord='fro'))

valid1 = np.floor(x_valid[:1000] * disc_data + 0.5)
valid1 = valid1.astype(int)
pr_valid = np.abs(np.matmul(valid1, Pr))

matrix_to_txt(np.transpose(Pr), 'testdata/mnist_mat_proj')
matrix_to_txt(np.transpose(Di3), 'testdata/mnist_mat_diag')
matrix_to_txt(valid1, 'testdata/mnist_x')
matrix_to_txt(pr_valid, 'testdata/mnist_x_proj')