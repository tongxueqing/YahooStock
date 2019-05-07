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
        995 nodes -> 500 nodes -> 199 nodes
    '''
    w1 = get_weight([input_num, hidden_num], regularizer)
    b1 = get_bias([hidden_num])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    # hidden layer calculation
    # X x W1 -> Z1; f(Z1) -> Y1
    # (749 * 995) x (995 * 500) -> (749 * 500) 

    w2 = get_weight([hidden_num, output_num], regularizer)
    b2 = get_bias([output_num])
    y = tf.matmul(y1, w2) + b2
    # output layer calculation
    # Y1 x W2 -> Y
    # (749 * 500) x (500 * 198) -> (749 * 199) 

    return y, w1, w2, b1, b2

def forward_2(x, y, input_num, output_num, regularizer, hidden_num = 700, hidden_num2 = 400):
    w1 = get_weight([input_num, hidden_num], regularizer)
    b1 = get_bias([hidden_num])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([hidden_num, hidden_num2], regularizer)
    b2 = get_bias([hidden_num2])
    y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)

    w3 = get_weight([hidden_num2, output_num], regularizer)
    b3 = get_bias([output_num])
    y = tf.matmul(y2, w3) + b3
    # output layer calculation
    # Y1 x W2 -> Y
    # (749 * 500) x (500 * 198) -> (749 * 199) 

    return y, w1, w2, b1, b2
