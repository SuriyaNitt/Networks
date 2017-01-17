import tflearn

def network(rows, cols):
    # Building Residual Network
    net = tflearn.input_data(shape=[None, rows, cols, 3])
    net = tflearn.conv_2d(net, 32, 3, activation='relu', bias=False)
    net = tflearn.max_pool_2d(net, 2)
    # Residual blocks
    net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 32, 128, downsample=True)
    #net = tflearn.residual_bottleneck(net, 2, 64, 256, downsample=True)
    #net = tflearn.residual_bottleneck(net, 2, 128, 512, downsample=True)

    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'prelu')
    net = tflearn.global_avg_pool(net)

    # Regression
    net = tflearn.fully_connected(net, 512, activation='prelu')
    net = tflearn.fully_connected(net, 1, activation='prelu')
    net = tflearn.regression(net, optimizer='momentum',
                             loss='mean_square',
                             learning_rate=0.1)
    return net

def double_network(rows, cols):
    # Building Residual Network
    net1 = tflearn.input_data(shape=[None, rows, cols, 3])
    net1 = tflearn.conv_2d(net1, 32, 3, activation='relu', bias=False)
    net1 = tflearn.max_pool_2d(net1, 2)
    # Residual blocks
    net1 = tflearn.residual_bottleneck(net1, 2, 16, 64, downsample=True)
    net1 = tflearn.residual_bottleneck(net1, 2, 32, 128, downsample=True)

    net1 = tflearn.batch_normalization(net1)
    net1 = tflearn.activation(net1, 'prelu')
    net1 = tflearn.global_avg_pool(net1)


    # Building Residual Network
    net2 = tflearn.input_data(shape=[None, rows, cols, 3])
    net2 = tflearn.conv_2d(net2, 32, 3, activation='relu', bias=False)
    net2 = tflearn.max_pool_2d(net2, 2)
    # Residual blocks
    net2 = tflearn.residual_bottleneck(net2, 2, 16, 64, downsample=True)
    net2 = tflearn.residual_bottleneck(net2, 2, 32, 128, downsample=True)

    net2 = tflearn.batch_normalization(net2)
    net2 = tflearn.activation(net2, 'prelu')
    net2 = tflearn.global_avg_pool(net2)

    # Merge layers
    net = tflearn.merge([net1, net2], mode='concat')

    # Add LSTM layers
    net = tflearn.reshape(net, new_shape=[1, 32, 256])
    net = tflearn.lstm(net, 128, return_seq=True)
    net = tflearn.lstm(net, 128)

    # Regression
    net = tflearn.fully_connected(net, 512, activation='prelu')
    net = tflearn.fully_connected(net, 1, activation='prelu')
    net = tflearn.regression(net, optimizer='momentum',
                             loss='mean_square',
                             learning_rate=0.1)
    return net


def threeD_conv_net_two_streams(depth, rows, cols, batch_size):
    # Building Residual Network
    net1 = tflearn.input_data(shape=[None, depth, rows, cols, 3])
    net1 = tflearn.conv_3d(net1, 4, 3, activation='relu', bias=False)
    net1 = tflearn.max_pool_3d(net1, 2, strides=2)
    net1 = tflearn.conv_3d(net1, 8, 3, activation='relu', bias=False)
    net1 = tflearn.max_pool_3d(net1, 2, strides=2)
    net1 = tflearn.conv_3d(net1, 16, 3, activation='relu', bias=False)
    net1 = tflearn.max_pool_3d(net1, 2, strides=2)

    # Building Residual Network
    net2 = tflearn.input_data(shape=[None, depth, rows, cols, 3])
    net2 = tflearn.conv_3d(net2, 4, 3, activation='relu', bias=False)
    net2 = tflearn.max_pool_3d(net2, 2, strides=2)
    net2 = tflearn.conv_3d(net2, 8, 3, activation='relu', bias=False)
    net2 = tflearn.max_pool_3d(net2, 2, strides=2)
    net2 = tflearn.conv_3d(net2, 16, 3, activation='relu', bias=False)
    net2 = tflearn.max_pool_3d(net2, 2, strides=2)

    # Merge layers
    net = tflearn.merge([net1, net2], mode='elemwise_mul')
    #net = tflearn.flatten(net)

    # Add LSTM layers
    net = tflearn.reshape(net, new_shape=[batch_size, 4, 28*28*16])
    net = tflearn.lstm(net, 32, weights_init='Xavier')
    #net = tflearn.lstm(net, 128, weights_init='Xavier')
    #net = tflearn.lstm(net, 16, weights_init='Xavier')

    # Regression
    #net = tflearn.fully_connected(net, 128, activation='prelu')
    #leaky_relu = tflearn.leaky_relu(net, alpha=0.2)
    net = tflearn.fully_connected(net, 1, activation='prelu')
    #net = tflearn.activations.leaky_relu(net, alpha=0.2)
    net = tflearn.regression(net, optimizer='adam',
                             loss='mean_square',
                             learning_rate=0.1)
    return net

def threeD_conv_net_single_stream(depth, rows, cols, batch_size, lr, myMetric):
    # Building Residual Network
    net = tflearn.input_data(shape=[None, depth, rows, cols, 3])
    net = tflearn.conv_3d(net, 4, 3, activation='relu', bias=False)
    net = tflearn.max_pool_3d(net, 2, strides=2)
    net = tflearn.conv_3d(net, 8, 3, activation='relu', bias=False)
    net = tflearn.max_pool_3d(net, 2, strides=2)
    net = tflearn.conv_3d(net, 16, 3, activation='relu', bias=False)
    net = tflearn.max_pool_3d(net, 2, strides=2)

    # Add LSTM layers
    net = tflearn.reshape(net, new_shape=[batch_size, depth/8, 28*28*16])
    net = tflearn.lstm(net, 32, weights_init='Xavier')
    #net = tflearn.lstm(net, 128, weights_init='Xavier')
    #net = tflearn.lstm(net, 16, weights_init='Xavier')

    # Regression
    #net = tflearn.flatten(net)
    #net = tflearn.fully_connected(net, 128, activation='prelu')
    #leaky_relu = tflearn.leaky_relu(net, alpha=0.2)
    net = tflearn.fully_connected(net, 2, activation='prelu')
    #net = tflearn.activations.leaky_relu(net, alpha=0.2)
    net = tflearn.regression(net,  metric=myMetric, optimizer='adam',
                             loss='mean_square',
                             learning_rate=lr)
    return net
