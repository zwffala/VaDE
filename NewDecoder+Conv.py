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
        return 784, 700, 10, 0.002, 0.002, 10, 0.9, 0.9, 1, 'sigmoid'
    if dataset == 'reuters10k':
        return 2000, 15, 4, 0.002, 0.002, 5, 0.5, 0.5, 1, 'linear'
    if dataset == 'har':
        return 561, 120, 6, 0.002, 0.00002, 10, 0.9, 0.9, 5, 'linear'


def gmmpara_init():
    lambda_init = tf.abs(tf.truncated_normal(shape=(latent_dim, n_centroid), mean=1, stddev=0.5, dtype=tf.float32))

    lambda_p = tf.get_variable(name='lambda', dtype=tf.float32,
                               initializer=lambda_init, trainable=True)

    theta_p = tf.Variable(tf.ones(shape=(n_centroid), dtype=tf.float32) * (1 / n_centroid), name='pi')
    u_p = tf.get_variable(name='u_p', shape=[latent_dim, n_centroid],
                               initializer=tf.initializers.truncated_normal(mean=0, stddev=0.5),
                               dtype=tf.float32)
    #u_p = tf.get_variable(name='u_p', shape=(latent_dim, n_centroid), dtype=tf.float32,
    #                      initializer=tf.initializers.zeros, trainable=True)
    return theta_p, u_p, lambda_p


def get_gamma(tempz):
    # tempz.shape(batch, latent)
    temp_Z = tf.expand_dims(tempz, -1)
    temp_Z = tf.tile(temp_Z, [1, 1, n_centroid])

    temp_u_tensor3 = tf.expand_dims(u_p, 0)
    temp_u_tensor3 = tf.tile(temp_u_tensor3, [batch_size, 1, 1])

    temp_lambda_tensor3 = tf.expand_dims(lambda_p, 0)
    temp_lambda_tensor3 = tf.tile(temp_lambda_tensor3, [batch_size, 1, 1])

    temp_theta_tensor3 = tf.expand_dims(theta_p, 0)
    temp_theta_tensor3 = tf.tile(temp_theta_tensor3, [latent_dim, 1])
    temp_theta_tensor3 = tf.expand_dims(temp_theta_tensor3, 0)
    temp_theta_tensor3 = tf.tile(temp_theta_tensor3, [batch_size, 1, 1])

    temp_p_c_z = tf.exp(tf.reduce_sum((tf.log(temp_theta_tensor3) - 0.5 * tf.log(2 * math.pi * temp_lambda_tensor3)
                                       - tf.square(temp_Z - temp_u_tensor3) / (2 * temp_lambda_tensor3)),
                                      axis=1)) + 1e-10
    return temp_p_c_z / tf.reduce_sum(temp_p_c_z, axis=-1, keepdims=True)


def vae_loss(x, x_decoded_mean):
    Z = tf.expand_dims(z, -1)  # z.shape(batch_size, latent)
    Z = tf.tile(Z, [1, 1, n_centroid])  # Z.shape(batch_size, latent, K)

    Z_mean_t = tf.expand_dims(z_mean, -1)  # z_mean.shape(batch_size, latent)
    Z_mean_t = tf.tile(Z_mean_t, [1, 1, n_centroid])

    Z_log_var_t = tf.expand_dims(z_log_var, -1)
    Z_log_var_t = tf.tile(Z_log_var_t, [1, 1, n_centroid])  # Z_log_var_t.shape(batch, latent, K)

    u_tensor3 = tf.expand_dims(u_p, 0)  # u_tensor3.shape(latent, K)
    u_tensor3 = tf.tile(u_tensor3, [batch_size, 1, 1])  # u_tensor3.shape(batch_size, latent, K)

    lambda_tensor3 = tf.expand_dims(lambda_p, 0)
    lambda_tensor3 = tf.tile(lambda_tensor3, [batch_size, 1, 1])

    theta_tensor3 = tf.expand_dims(theta_p, 0)
    theta_tensor3 = tf.tile(theta_tensor3, [latent_dim, 1])
    theta_tensor3 = tf.expand_dims(theta_tensor3, 0)
    theta_tensor3 = tf.tile(theta_tensor3, [batch_size, 1, 1])

    p_c_z = tf.exp(tf.reduce_sum(
        tf.log(theta_tensor3) - 0.5 * tf.log(2 * math.pi * lambda_tensor3) - tf.square(Z - u_tensor3) / (
                    2 * lambda_tensor3), axis=1)) + 1e-10  # p_c_z.shape(batch, K)
    gamma = p_c_z / tf.reduce_sum(p_c_z, axis=-1, keepdims=True)
    gamma_t = tf.expand_dims(gamma, axis=1)
    gamma_t = tf.tile(gamma_t, [1, latent_dim, 1])  # gamma_t.shape(batch, latent, 1)

    latent_loss = tf.reduce_sum(0.5 * gamma_t * (latent_dim * tf.log(math.pi * 2) + tf.log(lambda_tensor3)
                                                 + tf.exp(Z_log_var_t) / lambda_tensor3 + tf.square(
                Z_mean_t - u_tensor3) / lambda_tensor3), axis=(1, 2)) \
                  - 0.5 * tf.reduce_sum(z_log_var + 1, axis=-1) \
                  - tf.reduce_sum(tf.log(tf.tile(tf.expand_dims(theta_p, 0), [batch_size, 1])) * gamma, axis=-1) \
                  + tf.reduce_sum(tf.log(gamma) * gamma, axis=-1)  # latent_loss.shape(batch, )
    latent_loss = tf.reduce_mean(latent_loss)
    _latent_loss_scalar = tf.summary.scalar('latent_loss', latent_loss)

    if datatype == 'sigmoid':
        # recon_loss = alpha * original_dim * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_decoded_mean), axis=-1)   # recon_loss.shape(batch, )
        recon_loss = alpha * tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_decoded_mean), axis=-1)
    else:
        recon_loss = alpha * original_dim * tf.losses.mean_squared_error(labels=x, predictions=x_decoded_mean)

    recon_loss = tf.reduce_mean(recon_loss)
    _recon_loss_scalar = tf.summary.scalar('recon_loss', recon_loss)
    _loss = recon_loss + latent_loss
    _loss_scalar = tf.summary.scalar('loss', _loss)
    # return _loss
    return _loss, _latent_loss_scalar, _recon_loss_scalar, _loss_scalar


# def epochBegin(epoch):
#
#     if epoch % decay_n == 0 and epoch!=0:
#         lr_decay()
#
#     sample = sample_output.predict(X,batch_size=batch_size)
#     g = mixture.GMM(n_components=n_centroid,covariance_type='diag')
#     g.fit(sample)
#     p=g.predict(sample)
#     acc_g=cluster_acc(p,Y)
#
#     if epoch <1 and ispretrain == False:
#         u_p.set_value(floatX(g.means_.T))
#         print ('no pretrain,random init!')
#
#     gamma = gamma_output.predict(X,batch_size=batch_size)
#     acc=cluster_acc(np.argmax(gamma,axis=1),Y)
#     global accuracy
#     accuracy+=[acc[0]]
#     if epoch>0 :
#         #print ('acc_gmm_on_z:%0.8f'%acc_g[0])
#         print ('acc_p_c_z:%0.8f'%acc[0])
#     if epoch==1 and dataset == 'har' and acc[0]<0.77:
#         print ('=========== HAR dataset:bad init!Please run again! ============')
#         sys.exit(0)
#
# class EpochBegin(Callback):
#     def on_epoch_begin(self, epoch, logs={}):
#         epochBegin(epoch)

'''
In decoding,  add mu_x and sigma_x, compute x’ through those
'''
def resolve_convolutional_variational_autoencoder(data, latent_dim, trainable=True):
    input_shape = (tf.shape(data)[0], 28, 28, 1)
    _input = tf.reshape(data, input_shape)

    filters = [32, 64, 128, latent_dim]

    with tf.name_scope("convolution_encoder"):
        cae_encoder = tf.layers.conv2d(_input, filters[0], 5, strides=(2, 2),
                                       padding='same', activation=tf.nn.relu)

        cae_encoder = tf.layers.conv2d(cae_encoder, filters[1], 5, strides=(2, 2),
                                       padding='same', activation=tf.nn.relu)
        cae_encoder = tf.layers.conv2d(cae_encoder, filters[2], 3, strides=(2, 2),
                                       padding='valid', activation=tf.nn.relu)
    dim_before_flatten = tf.shape(cae_encoder)
    print('dim_before_flatten: ', dim_before_flatten)
    cae_encoder = tf.layers.Flatten()(cae_encoder)
    flatten_dim = filters[2] * int(input_shape[1] / 8) * int(input_shape[1] / 8)

    mu_layer = tf.layers.Dense(units=latent_dim,
                               trainable=trainable,
                               activation=None,
                               name='mu_layer')
    print('cae_encoder: ', cae_encoder)
    z_mu = mu_layer.apply(cae_encoder)

    sigma_layer = tf.layers.Dense(units=latent_dim,
                                  trainable=trainable,
                                  activation=None,
                                  name='sigma_layer')
    z_sigma = sigma_layer.apply(cae_encoder)

    eps = tf.random_normal(shape=tf.shape(z_sigma), mean=0, stddev=1, dtype=tf.float32)  # eps.shape(batch_size, latent)
    z = z_mu + tf.exp(z_sigma / 2) * eps  # z.shape(batch_size, latent)

    gamma = get_gamma(z)

    with tf.name_scope("convolution_decoder"):
        cae_decoder = tf.layers.dense(z, flatten_dim, activation=tf.nn.relu)
        cae_decoder = tf.reshape(cae_decoder, shape=dim_before_flatten)
        print('cae_decoder1: ', cae_decoder)
        cae_decoder = tf.layers.Conv2DTranspose(filters[1], 3, strides=(2, 2),
                                                padding='valid', activation=tf.nn.relu)(cae_decoder)
        cae_decoder = tf.layers.Conv2DTranspose(filters[0], 5, strides=(2, 2),
                                                padding='same', activation=tf.nn.relu)(cae_decoder)
        cae_decoder = tf.layers.Conv2DTranspose(input_shape[-1], 5, strides=(2, 2),
                                                padding='same', activation=None)(cae_decoder)
        print('cae_decoder2: ', cae_decoder)
        cae_decoder = tf.layers.Flatten()(cae_decoder)

        output_mu_layer = tf.layers.Dense(units=28*28*1, trainable=trainable, activation=None, name='output_mu')
        mu_x = output_mu_layer.apply(cae_decoder)

        output_sigma_layer = tf.layers.Dense(units=28*28*1, trainable=trainable, activation=None, name='output_sigma')
        sigma_x = output_sigma_layer.apply(cae_decoder)

        eps_x = tf.random_normal(shape=tf.shape(28*28*1), mean=0, stddev=1, dtype=tf.float32)
        output = mu_x + tf.exp(sigma_x/2)*eps_x

    output = tf.reshape(output, shape=tf.shape(data))

    return data, z_mu, z_sigma, z, gamma, output


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

theta_p, u_p, lambda_p = gmmpara_init()
x, z_mean, z_log_var, z, tempGamma, x_decoded_mean = resolve_convolutional_variational_autoencoder(data, latent_dim)
#x, encoder_layers, z_mean, z_log_var, z, tempGamma, decoder_layers, x_decoded_mean = resolve_variational_autoencoder(data, original_dim, intermediate_dim, latent_dim, datatype)

loss, latent_loss_scalar, recon_loss_scalar, loss_scalar = vae_loss(x, x_decoded_mean)
acc_tensor = tf.placeholder(dtype=tf.float32, shape=(), name='acc')
acc = tf.summary.scalar('acc_p_c_z', acc_tensor)
merged_acc_op = tf.summary.merge([acc])
merged_training_summary = tf.summary.merge([latent_loss_scalar, recon_loss_scalar, loss_scalar])

global_step = tf.Variable(0, trainable=False)
# learning_rate = tf.math.maximum(tf.train.exponential_decay(lr_nn, global_step, 7000, 0.9), 0.0002)
learning_rate = tf.train.exponential_decay(0.002, global_step, 1000, 0.95)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam_Optimizer').minimize(loss,
                                                                                                global_step=global_step)
init_param = tf.global_variables_initializer()

n_train = 70000
training_batch_size = 1024
n_batches = int(n_train / training_batch_size)
summaryDir = './'
saver = tf.train.Saver()

pretrain = False
modelDir = './model/model.ckpt'


def reinitialzeGMMVariables(sample):
    g = mixture.GaussianMixture(n_components=n_centroid, covariance_type='diag')
    g.fit(sample)
    up_assign_op = u_p.assign(g.means_.T)
    lambda_assign_op = lambda_p.assign(g.covariances_.T)
    return up_assign_op, lambda_assign_op


with tf.Session() as sess:
    sess.run(init_param)
    if pretrain:
        # from tensorflow.python.tools import inspect_checkpoint as chkp
        # chkp.print_tensors_in_checkpoint_file(modelDir, tensor_name='', all_tensors=True)
        saver.restore(sess, modelDir)
        print('up after restore: ', u_p.eval(sess))
        # sample, _global_step = sess.run([z_mean, global_step.assign(0)], feed_dict={data: X, batch_size: n_train})
        sample = sess.run(z_mean, feed_dict={data: X, batch_size: n_train})
        print('sample.shape: ', sample.shape)
        print('sample: ', sample[:3])
        up_assign_op, lambda_assign_op = reinitialzeGMMVariables(sample)
        sess.run([up_assign_op, lambda_assign_op, global_step.assign(0)])
        print('up after reinitial: ', u_p.eval(sess))
        print('lambda_p after reinitial: ', lambda_p.eval(sess))
        g = mixture.GaussianMixture(n_components=n_centroid, covariance_type='diag')
        g.fit(sample)
        p = g.predict(sample)
        acc_g = cluster_acc(p, Y)
        print('acc_g: ', acc_g[0])
    aidx = list(range(n_train))
    writer = tf.summary.FileWriter(summaryDir, sess.graph)
    for i in range(epoch):
        random.shuffle(aidx)
        ptr = 0
        #if i == 0:
            # initialize u_p with training scope GMM
            #sample = sess.run(z_mean, feed_dict={data: X, batch_size: n_train})
            #g = mixture.GaussianMixture(n_components=n_centroid, covariance_type='diag')
            #g.fit(sample)
            #up_assign_op = u_p.assign(g.means_.T)
            #sess.run(up_assign_op)
            #print('u_p after initializaition: ', u_p.eval(sess))
        for j in range(n_batches):
            inp = X[aidx[ptr:ptr + training_batch_size], :]
            ptr += training_batch_size
            _, _ce, _lr, summary = sess.run([optimizer, loss, learning_rate, merged_training_summary],
                                            feed_dict={data: inp, label: inp, batch_size: training_batch_size})
            progress(j + 1, n_batches, status=' Loss=%f, Lr=%f, Epoch=%d/%d' % (_ce, _lr, i + 1, epoch))
            writer.add_summary(summary, i * n_batches + j)
        cluster_prec = sess.run(tempGamma, feed_dict={data: X, batch_size: n_train})
        acc = cluster_acc(np.argmax(cluster_prec, axis=1), Y)
        print('acc_p_c_z: ', acc[0])
        acc_summary = sess.run(merged_acc_op, feed_dict={acc_tensor: acc[0]})
        writer.add_summary(acc_summary, i)
    save_path = saver.save(sess, "./model/model.ckpt")
    print("Model saved in path: %s" % save_path)
