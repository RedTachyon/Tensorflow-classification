import tensorflow as tf


class LogisticGraph:
    """
    Implements a simple multiclass logistic regression model.
    """

    def __init__(self, n_features, n_categories):
        self.N = self.n_features = n_features
        self.C = self.n_categories = n_categories

        self._build_graph()

    def _build_graph(self):
        """
        Builds the computation graph for the logistic regression model.
        """
        self.X = tf.placeholder(tf.float32, [self.N, None])
        self.y = tf.placeholder(tf.float32, [self.C, None])

        self.W = tf.get_variable("W", shape=[self.C, self.N], initializer=tf.truncated_normal_initializer)
        self.b = tf.get_variable("b", shape=[self.C, 1], initializer=tf.zeros_initializer)

        self.z = tf.matmul(self.W, self.X) + self.b
        self.y_hat = tf.nn.softmax(self.z, dim=0)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.z, dim=0))

        self.train = tf.train.AdamOptimizer().minimize(self.loss)

        self.correct_pred = tf.equal(tf.argmax(self.y, 0), tf.argmax(self.y_hat, 0))

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.model = tf.global_variables_initializer()


class TwoLayerNN:
    """
    Implements a two-layer neural network.
    """

    def __init__(self, n_features, n_categories, num_hidden, activation='relu', beta=0):
        self.N = self.n_features = n_features
        self.C = self.n_categories = n_categories
        self.h = self.num_hidden = num_hidden

        self.beta = beta

        activations = {
            'relu': tf.nn.relu,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid
        }

        self.activation = activations[activation]

        self._build_graph()

    def _build_graph(self):
        """
        Builds the computation graph for the two-layer model.
        """
        self.X = tf.placeholder(tf.float32, [self.N, None])
        self.y = tf.placeholder(tf.float32, [self.C, None])

        self.W1 = tf.get_variable("W1", shape=[self.h, self.N], initializer=tf.truncated_normal_initializer)
        self.b1 = tf.get_variable("b1", shape=[self.h, 1], initializer=tf.zeros_initializer)

        self.W2 = tf.get_variable("W2", shape=[self.C, self.h], initializer=tf.truncated_normal_initializer)
        self.b2 = tf.get_variable("b2", shape=[self.C, 1], initializer=tf.truncated_normal_initializer)

        self.z1 = tf.matmul(self.W1, self.X) + self.b1
        self.a1 = self.activation(self.z1)

        self.z2 = tf.matmul(self.W2, self.a1) + self.b2
        self.y_hat = tf.nn.softmax(self.z2, dim=0)

        self.l2_reg = tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.z2, dim=0)) \
                    + self.beta * self.l2_reg

        self.train = tf.train.AdamOptimizer().minimize(self.loss)

        self.correct_pred = tf.equal(tf.argmax(self.y, 0), tf.argmax(self.y_hat, 0))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.model = tf.global_variables_initializer()


class KNNGraph:
    """
    Implements a KNN classifier.
    """

    def __init__(self, n_features, n_categories, K):
        self.N = self.n_features = n_features
        self.C = self.n_categories = n_categories
        self.K = K

        self._build_graph()

    def _build_graph(self):
        """
        Builds the computation graph for the KNN model.
        """
        self.X_tr = tf.placeholder(tf.float32, [self.N, None])
        self.Y_tr = tf.placeholder(tf.float32, [self.C, None])
        self.X_te = tf.placeholder(tf.float32, [self.N, 1])

        self.distance = tf.negative(tf.sqrt(tf.reduce_sum(tf.square(self.X_tr - self.X_te), reduction_indices=0)))
        self.values, self.indices = tf.nn.top_k(self.distance, k=self.K, sorted=False)

        self.nearest_neighbors = []
        for i in range(self.K):
            self.nearest_neighbors.append(tf.argmax(self.Y_tr[:, self.indices[i]]))

        self.neighbours_tensor = tf.stack(self.nearest_neighbors)

        self.y, self.idx, self.count = tf.unique_with_counts(self.neighbours_tensor)
        self.pred = tf.slice(self.y, begin=[tf.argmax(self.count, 0)], size=tf.constant([1], dtype=tf.int64))[0]
