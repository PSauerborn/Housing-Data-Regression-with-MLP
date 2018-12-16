from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

data = pd.read_csv('./housing.data',
                   header=None, sep='\s+', dtype=float)

data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

target = data.MEDV
data = data.drop(columns=['MEDV'])


def generate_data_sets(data):

    X_train, X_test = data

    s = StandardScaler()
    n = Normalizer()

    X_train_std = s.fit_transform(X_train)
    X_test_std = s.transform(X_test)

    X_train_norm = n.fit_transform(X_train)
    X_test_norm = n.transform(X_test)

    norm = (X_train_norm, X_test_norm)
    std = (X_train_std, X_test_std)

    return std, norm


X_train, X_test, y_train, y_test = train_test_split(data.values, target.values, test_size=0.1, random_state=1)
std, norm = generate_data_sets((X_train, X_test))

X_train_std, X_test_std = std
X_train_norm, X_test_norm = norm


class MLPClassifier():

    def __init__(self, random_state, learning_rate):

        self.random_state = random_state
        self.rgen = np.random.RandomState(random_state)
        self.learning_rate = learning_rate
        self.g = tf.Graph()

    def build(self, n_inputs):

        with self.g.as_default():

            tf.set_random_seed(self.random_state)

            tf_x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='tf_x')
            tf_y = tf.placeholder(tf.float32, shape=(None, 1), name='tf_y')
            eta = tf.placeholder(tf.float32, name='eta')
            keep_proba = tf.placeholder(tf.float32, name='keep_proba')
            tf_lambda = tf.placeholder(tf.float32, name='lambda')

            h1 = tf.layers.dense(tf_x, units=128, activation=tf.nn.relu, name='h1')
            h1 = tf.nn.dropout(h1, keep_proba)


            h2 = tf.layers.dense(h1, units=264, activation=tf.nn.sigmoid, name='h2')
            h2 = tf.nn.dropout(h2, keep_proba)

            output = tf.layers.dense(h2, units=1, activation=None, name='output')

            cost = tf.losses.mean_squared_error(tf_y, output)

            self.train_op = tf.train.GradientDescentOptimizer(eta).minimize(cost)

            self.init_op = tf.global_variables_initializer()

    def train(self, X, y, n_epochs, X_test, eta):

        with tf.Session(graph=self.g) as sess:

            sess.run(self.init_op)

            cost_avg = []

            for epoch in range(1, n_epochs + 1):


                feed = {'tf_x:0': X, 'tf_y:0': y, 'eta:0': eta,
                        'keep_proba:0': 0.5, 'lambda:0': 0.1}

                c, _ = sess.run(['mean_squared_error/value:0', self.train_op], feed_dict=feed)

                cost_avg.append(c)

                print('Epoch: {} Cost: {:.2f}'.format(epoch, c))

            y_pred = sess.run('output/BiasAdd:0', feed_dict={'tf_x:0': X_test, 'keep_proba:0': 1, 'lambda:0': 0.1})

        return cost_avg, y_pred

n_epochs = 200
n_inputs = X_train_std.shape[1]

mlp = MLPClassifier(random_state=1, learning_rate=0.01)
mlp.build(n_inputs)

cost, y_pred = mlp.train(X_train_std, y_train.reshape(-1, 1), n_epochs, X_test_std, eta=0.01)

y = pd.DataFrame(y_test, columns=['true'])
y['predicted'] = y_pred
y['percentage error'] = abs(1 - (y.iloc[:,0] / y.iloc[:, 1]))*100

print(y)

average_error = np.mean(y['percentage error'])
print('Average Percentage Error: {:.2f}%'.format(average_error))


fig, ax = plt.subplots()
ax.plot(cost)
plt.show()
