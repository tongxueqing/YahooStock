import tensorflow as tf
import numpy as np
import NN_input
import NN_forward

STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001
LEANING_RATE_DECAY = 0.999
REGULARIZER = 0.01

def backward():
    '''
    backpropagation function
    learn the weight matrices (w1, w2) 
    '''
    X, Y = NN_input.get_data()
    x = tf.placeholder(tf.float32, shape = (None, X.shape[1]))
    y = tf.placeholder(tf.float32, shape = (None, Y.shape[1]))
    # make fake inputs to build the pipeline and add the real input at last

    y_hat = NN_forward.forward(x, REGULARIZER)
    # estimation of y

    global_step = tf.Variable(0, trainable = False)
    # assign the STEP counter
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            X.shape[0] / BATCH_SIZE
            staircase = True)
    # set the decay of learning rate

    loss_mse = tf.reduce_mean(tf.square(y_hat - y))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))
    # mean of square error
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % X.shape
            end = 
            sess.run(train_step, feed_dict = {x: X[start:end], y = Y[start:end]})
