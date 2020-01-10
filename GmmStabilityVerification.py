# -*- coding: utf-8 -*-
'''
VaDE (Variational Deep Embedding:A Generative Approach to Clustering)

Best clustering accuracy:
MNIST: 94.46% +
Reuters10k: 81.66% +
HHAR: 85.38% +
Reuters_all: 79.38% +

@code author: Zhuxi Jiang
'''
import numpy as np
import scipy.io as scio
import gzip
from six.moves import cPickle
import random
import math

# import warnings
# warnings.filterwarnings("ignore")

import tensorflow as tf
import sys

from sklearn import mixture

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



'''
using Vae to get z and do gmm on the same z several times to get accuracy difference
'''
'''
using k-means for initialize u_c
'''


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    if count == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, w


def load_data(dataset):
    path = 'dataset/' + dataset + '/'
    if dataset == 'mnist':
        path = path + 'mnist.pkl.gz'
        if path.endswith(".gz"):
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'rb')

        if sys.version_info < (3,):
            (x_train, y_train), (x_test, y_test) = cPickle.load(f)
        else:
            (x_train, y_train), (x_test, y_test) = cPickle.load(f, encoding="bytes")

        f.close()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        X = np.concatenate((x_train, x_test))
        Y = np.concatenate((y_train, y_test))

    if dataset == 'reuters10k':
        data = scio.loadmat(path + 'reuters10k.mat')
        X = data['X']
        Y = data['Y'].squeeze()

    if dataset == 'har':
        data = scio.loadmat(path + 'HAR.mat')
        X = data['X']
        X = X.astype('float32')
        Y = data['Y'] - 1
        X = X[:10200]
        Y = Y[:10200]

    return X, Y


def config_init(dataset):
    if dataset == 'mnist':
        return 784, 150, 10, 0.002, 0.002, 10, 0.9, 0.9, 1, 'sigmoid'
    if dataset == 'reuters10k':
        return 2000, 15, 4, 0.002, 0.002, 5, 0.5, 0.5, 1, 'linear'
    if dataset == 'har':
        return 561, 120, 6, 0.002, 0.00002, 10, 0.9, 0.9, 5, 'linear'




def loss(x, x_decoded_mean, z_mu, z_sigma, alpha):
    _recon_loss = alpha * tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_decoded_mean), axis=-1)
    _recon_loss = tf.reduce_mean(_recon_loss)
    _recon_loss_scalar = tf.summary.scalar('recon_loss', _recon_loss)
    _latent_loss = tf.reduce_sum(
        tf.square(z_mu) + tf.square(z_sigma) - tf.log(1e-10 + tf.square(z_sigma)) - 1, axis=-1)
    _latent_loss = tf.reduce_mean(_latent_loss)
    _latent_loss_scalar = tf.summary.scalar('latent_loss', _latent_loss)
    _loss = _recon_loss + _latent_loss
    _loss_scalar = tf.summary.scalar('loss', _loss)
    return _loss, _recon_loss_scalar, _latent_loss_scalar, _loss_scalar

'''
A `Tensor` which represents input layer of a model. Its shape
    is (batch_size, first_layer_dimension) and its dtype is `float32`.
    first_layer_dimension is determined based on given `feature_columns`.
'''
def resolve_variational_autoencoder(data, original_dim, intermediate_dim, latent_dim, trainable=True):
    # construct the encoder dense layers
    # _input = tf.placeholder(shape=(None, original_dim), name='input', dtype=tf.float32)
    print('data shape:', data.shape)
    _input = data
    result = _input
    encoder_layers = []
    for i in intermediate_dim:
        _layer = tf.layers.Dense(units=i,
                                 activation='relu',
                                 trainable=trainable)

        result = _layer.apply(result)
        encoder_layers.append(_layer)

    # bottleneck layer, i.e. features are extracted from here

    mu_layer = tf.layers.Dense(units=latent_dim,
                               trainable=trainable,
                               activation=None)
    encoder_layers.append(mu_layer)
    z_mu = mu_layer.apply(result)

    sigma_layer = tf.layers.Dense(units=latent_dim,
                                  trainable=trainable,
                                  activation=None)
    encoder_layers.append(sigma_layer)
    z_sigma = sigma_layer.apply(result)

    eps = tf.random_normal(shape=tf.shape(z_sigma), mean=0, stddev=1, dtype=tf.float32)    # eps.shape(batch_size, latent)
    z = z_mu + tf.exp(z_sigma/2) * eps  # z.shape(batch_size, latent)
    result = z

    # construct the decoder dense layers
    decoder_layers = []
    for i in reversed(intermediate_dim):
        _layer = tf.layers.Dense(units=i,
                                 activation='relu',
                                 trainable=trainable)
        result = _layer.apply(result)
        decoder_layers.append(_layer)

    # construct the output layer
    _layer = tf.layers.Dense(units=original_dim,
                             trainable=trainable,
                             activation=None)

    decoder_layers.append(_layer)
    result = _layer.apply(result)

    return _input, z, z_mu, z_sigma, result


dataset = 'mnist'
db = sys.argv[1]
if db in ['mnist', 'reuters10k', 'har']:
    dataset = db
print('training on: ' + dataset)
batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
latent_dim = 10
intermediate_dim = [500, 500, 2000]
accuracy = []
X, Y = load_data(dataset)
original_dim, epoch, n_centroid, lr_nn, lr_gmm, decay_n, decay_nn, decay_gmm, alpha, datatype = config_init(dataset)
data = tf.placeholder(tf.float32, shape=(None, original_dim), name='data')
label = tf.placeholder(tf.float32, shape=(None, original_dim), name='label')

x, z, z_mu, z_sigma, x_decoded_mean = resolve_variational_autoencoder(data, original_dim, intermediate_dim, latent_dim)
#x, encoder_layers, z_mean, z_log_var, z, tempGamma, decoder_layers, x_decoded_mean = resolve_variational_autoencoder(data, original_dim, intermediate_dim, latent_dim, datatype)

loss, recon_loss_scalar, latent_loss_scalar, loss_scalar = loss(x, x_decoded_mean, z_mu, z_sigma, alpha)
acc_tensor = tf.placeholder(dtype=tf.float32, shape=(), name='acc')
acc = tf.summary.scalar('acc_p_c_z', acc_tensor)
merged_acc_op = tf.summary.merge([acc])
merged_training_summary = tf.summary.merge([recon_loss_scalar, latent_loss_scalar, loss_scalar])

global_step = tf.Variable(0, trainable=False)
# learning_rate = tf.math.maximum(tf.train.exponential_decay(lr_nn, global_step, 7000, 0.9), 0.0002)
learning_rate = tf.train.exponential_decay(0.002, global_step, 1000, 0.95)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam_Optimizer').minimize(loss,
                                                                                                global_step=global_step)
init_param = tf.global_variables_initializer()

n_train = 70000
training_batch_size = 256
n_batches = int(n_train / training_batch_size)
summaryDir = './'
saver = tf.train.Saver()

pretrain = False
modelDir = './model/model.ckpt'



with tf.Session() as sess:
    sess.run(init_param)
    aidx = list(range(n_train))
    writer = tf.summary.FileWriter(summaryDir, sess.graph)
    for i in range(epoch):
        random.shuffle(aidx)
        ptr = 0
        for j in range(n_batches):
            inp = X[aidx[ptr:ptr + training_batch_size], :]
            ptr += training_batch_size
            _, _ce, _lr, summary = sess.run([optimizer, loss, learning_rate, merged_training_summary],
                                            feed_dict={data: inp, label: inp, batch_size: training_batch_size})
            progress(j + 1, n_batches, status=' Loss=%f, Lr=%f, Epoch=%d/%d' % (_ce, _lr, i + 1, epoch))
            writer.add_summary(summary, i * n_batches + j)
    latent = sess.run(z, feed_dict={data: X, batch_size: n_train})

    accPredictTimes = 10
    for i in range(accPredictTimes):
        g = mixture.GaussianMixture(n_components=n_centroid, covariance_type='diag')
        g.fit(latent)
        p = g.predict(latent)
        mu = g.means_.T
        sigma = g.covariances_.T
        acc = cluster_acc(p, Y)
        print('means: ', mu)
        print('sigma: ', sigma)
        print('acc: ', acc[0])
