import tensorflow as tf
# popular neural network python package

def get_weight(shape, regularizer):
    '''
    The randomly initialized weight matrix of Neuro Network
    Given the shape value
    generate for each layer
    '''
    w = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w)) # add w matrix to regularizer: Regularization is to prevent overfitting
    return w

def get_bias(shape):
    '''
    get the randomly initialized bias value
    Usually, bias will not be regularized
    '''
    b = tf.Variable(tf.constant(0.01, shape = shape))
    return b

def forward(x, y, input_num, output_num, regularizer, hidden_num = 500):
    '''
    NN feed forward part
    first try:
        input        hidden       output
        945 nodes -> 500 nodes -> 199 nodes
    '''
    w1 = get_weight([input_num, hidden_num], regularizer)
    b1 = get_bias([hidden_num])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    # hidden layer calculation

    w2 = get_weight([hidden_num, output_num], regularizer)
    b2 = get_bias([output_num])
    y = tf.matmul(y1, w2) + b2
    # output layer calculation
    return y
