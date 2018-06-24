import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import prepare_stocks_dataset as prep
import clean_stocks_dataset as cln
import os.path
import sys

def predictor(fname):

    # clean the dataset file and prepare the feature vectors
    data = clean_and_prepare(fname)
    X_train, y_train, X_test, y_test = get_data_split(data)         # split dataset into 80% training and 20% testing data
    X, Y, out, opt, sse = build_computation_graph(data)             # build the network

    # Make a tensorflow Session
    net = tf.Session()

    # Initialize all vairables defined with tesorflow
    net.run(tf.global_variables_initializer())

    # Setup interactive plot for showing progress
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(y_test)       # shows the true values of S&P index returns
    line2, = ax1.plot(y_test * 1.4) # shows the predicted values
    plt.show()

    # Number of epochs and batch size to train the neural network
    epochs = 10
    batch_size = 256

    for e in range(epochs):

        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        # Training network using batches of 256 records
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})

            # Show progress
            if np.mod(i, 25) == 0:
                # Prediction
                pred = net.run(out, feed_dict={X: X_test})
                line2.set_ydata(pred)
                plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
                file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.png'
                # plt.savefig(file_name)
                plt.pause(0.01)

    # Prediction
    pred = net.run(out, feed_dict={X: X_test})
    line2.set_ydata(pred)
    plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
    plt.xlabel('Minutes')
    plt.ylabel('S&P Returns - Percent Change')
    file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.png'
    plt.savefig(file_name)

    # Print final SSE after Training
    sse_final = net.run(sse, feed_dict={X: X_test, Y: y_test})

    # plot the difference between predicted and actual S&P returns
    plt.figure(2)
    plt.ioff()
    plt.scatter(pred, y_test, color='blue', label='Data')
    plt.xlabel('Predicted S&P Returns')
    plt.xlabel('True S&P Returns')
    plt.show()
    print('SSE : ', sse_final)

def clean_and_prepare(fname):
    # clean original file and store the clean data in a new file 'data_stocks_clean.csv'

    base_name = fname.split(",")[0]
    if not os.path.exists(fname):
        print('Please make sure the dataset file is in the same directory as code or provide the absolute path')
        sys.exit(0)

    # check if data preparation is already complete
    if not os.path.exists(base_name + '_prepared.csv'):
        # clean dataset and compute new features to be fed to Neural Network
        data = prep.get_features(fname)
    else:
        data = pd.read_csv(base_name + '_prepared.csv')

    return data

def get_data_split(data):

    rows = data.shape[0]   # rows
    cols = data.shape[1]   # columns

    data = data.values  # .values contains data as a Numpy Array

    # Training and test data
    train_start = 0
    train_end = int(np.floor(0.8 * rows))   # 80% of data is training data
    # test_start = train_end + 1            # remaining is for testing
    test_start = rows - 100
    test_end = rows
    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]

    # Scale data to handle outliers
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    # Build X and y
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]

    return X_train, y_train, X_test, y_test

def build_computation_graph(data):

    # Define Placeholders for passing input to network
    attributes = data.shape[1]
    n_stocks = attributes - 1  # number of input columns/features
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])  # None means we can give any number of rows
    Y = tf.placeholder(dtype=tf.float32, shape=[None])

    # Randomly initialize weights and biases for the neural network
    sigma = 1

    # use a uniform distribution around mean = 0
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    # Model architecture parameters
    n_neurons_1 = 1024
    n_neurons_2 = 512
    n_neurons_3 = 256
    n_neurons_4 = 128
    n_neurons_5 = 64
    n_target = 1

    # define input and output weights as variables at each layer
    # weights and biases will be updated by the network as it learns, so we define it as variables

    # Layer 1:
    W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

    # Layer 2:
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

    # Layer 3:
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

    # Layer 4:
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

    # Layer 5:
    W_hidden_5 = tf.Variable(weight_initializer([n_neurons_4, n_neurons_5]))
    bias_hidden_5 = tf.Variable(bias_initializer([n_neurons_5]))

    # Output layer:
    W_out = tf.Variable(weight_initializer([n_neurons_5, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))

    # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
    hidden_5 = tf.nn.relu(tf.add(tf.matmul(hidden_4, W_hidden_5), bias_hidden_5))

    # Output layer (must be transposed)
    out = tf.transpose(tf.add(tf.matmul(hidden_5, W_out), bias_out))

    # Cost function = Sum of Squared Errors
    # mse = tf.reduce_mean(tf.squared_difference(out, Y))
    sse = tf.reduce_sum(tf.squared_difference(out, Y))

    # Optimizer - AdamOptimizer - Advanced version of Gradient Descent optimizer
    opt = tf.train.AdamOptimizer().minimize(sse)

    return X, Y, out, opt, sse


if __name__ == '__main__':

    if len(sys.argv) != 1:
        print('Usage: \npython script_name.py <dataset-file>\n')
        sys.exit(0)

    ds_fname = sys.argv[1]
    predictor(ds_fname)

