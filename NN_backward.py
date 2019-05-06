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
    X, Y, sample_num, input_num, output_num = NN_input.get_data()
    x = tf.placeholder(tf.float32, shape = (None, input_num))
    y = tf.placeholder(tf.float32, shape = (None, output_num))
    # make fake inputs to build the pipeline and add the real input at last

    y_hat = NN_forward.forward(x, REGULARIZER)
    # estimation of y

    global_step = tf.Variable(0, trainable = False)
    # assign the STEP counter
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            sample_num / BATCH_SIZE
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
            start = (i * BATCH_SIZE) % sample_num
            end = 
            sess.run(train_step, feed_dict = {x: X[start:end], y = Y[start:end]})
            if i % 2000 == 0:
                loss_v = sess.run(loss_total, feed_dict = {x: X, y: Y})
                print('After %d steps, loss is %.2f' % (i, loss_v))
        y_estimate = sess.run(y_hat, feed_dict = {x: X_test}) 


if __name__ == '__main__':
    backward()
