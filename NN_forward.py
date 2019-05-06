import tensorflow as tf
# popular neural network python package

def get_weight(shape, regularizer):
    '''
    The randomly initialized weight matrix of Neuro Network
    Given the shape value
    generate for each layer
    '''
    w = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.12_regularizer(regularizer(w))) # add w matrix to regularizer: Regularization is to prevent overfitting
    return w

def get_bias(shape):
    '''
    get the randomly initialized bias value
    Usually, bias will not be regularized
    '''
    b = tf.Variable(tf.constant(0.01, shape = shape))
    return b

def forward(x, regularizer):
    '''

    '''
