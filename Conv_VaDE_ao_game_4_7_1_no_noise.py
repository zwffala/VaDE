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

import json
import json_lines

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


def load_data():
    X = []
    with open('dataset/ccss/ccss_318_414_multi_cuid.json', 'r') as rfile:
        for line in json_lines.reader(rfile):
            X.append(line['ts_game_start_counts_last_30_days_buckets'])
    X = np.array(X)
    # ln_n_plus data normalization
    X = np.log(X+1)
    print('shape of X ', X.shape)
    return X, len(X)


def config_init(dataset):
    if dataset == 'ao':
        return 28, 350, 7, 3, 'linear'


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

    latent_loss = tf.reduce_sum(0.5 * gamma_t * (latent_dim * tf.log(math.pi * 2) + tf.log(lambda_tensor3 + 1e-10)
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
    return _loss, recon_loss, latent_loss, _latent_loss_scalar, _recon_loss_scalar, _loss_scalar


def resolve_convolutional_variational_autoencoder2(data, latent_dim, trainable=True):
    input_shape = (tf.shape(data)[0], 4, 7, 1)
    _input = tf.reshape(data, input_shape)

    with tf.name_scope("convolution_encoder"):
        cnn1 = tf.layers.conv2d(_input, filters=1, kernel_size=(4, 1), strides=(1, 1), padding='valid', activation=tf.nn.elu)
        print('cnn1: ', cnn1)

        cnn2 = tf.layers.conv2d(cnn1, filters=1, kernel_size=(1, 3), strides=(1, 2), padding='valid', activation=tf.nn.elu)
        print('cnn2: ', cnn2)

        dim_before_flatten = tf.shape(cnn2)
        cae_encoder = tf.layers.Flatten()(cnn2)
        # check if it is correct shape
        print('cae_encoder: ', cae_encoder.shape)

        mu_layer = tf.layers.Dense(units=latent_dim,
                                   trainable=trainable,
                                   activation=None,
                                   name='mu_layer')
        z_mu = mu_layer.apply(cae_encoder)

        sigma_layer = tf.layers.Dense(units=latent_dim,
                                      trainable=trainable,
                                      activation=None,
                                      name='sigma_layer')
        z_sigma = sigma_layer.apply(cae_encoder)

        eps = tf.random_normal(shape=tf.shape(z_sigma), mean=0, stddev=1,
                               dtype=tf.float32)  # eps.shape(batch_size, latent)
        z = z_mu + tf.exp(z_sigma / 2) * eps  # z.shape(batch_size, latent)

        gamma = get_gamma(z)

        with tf.name_scope("convolution_decoder"):
            cae_decoder = tf.layers.dense(z, cae_encoder.shape[1], activation=tf.nn.elu)
            cae_decoder = tf.reshape(cae_decoder, shape=dim_before_flatten)
            print('cae_decoder1: ', cae_decoder)

            cae_decoder = tf.layers.Conv2DTranspose(filters=1, kernel_size=(1, 3), strides=(1, 2), padding='valid', activation=tf.nn.elu)(cae_decoder)
            print('cae_decoder2: ', cae_decoder)

            cae_decoder = tf.layers.Conv2DTranspose(filters=1, kernel_size=(4, 1), strides=(1, 1), padding='valid', activation=None)(cae_decoder)
            print('cae_decoder3: ', cae_decoder)

        output = tf.reshape(cae_decoder, shape=tf.shape(data))
        print('output: ', output)

        return data, z_mu, z_sigma, z, gamma, output


print('training on: ' + 'ao')
batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
X, n_train = load_data()
original_dim, epoch, n_centroid, alpha, datatype = config_init('ao')
latent_dim = 7
data = tf.placeholder(tf.float32, shape=(None, original_dim), name='data')

theta_p, u_p, lambda_p = gmmpara_init()
_, z_mean, z_log_var, z, tempGamma, x_decoded_mean = resolve_convolutional_variational_autoencoder2(data, latent_dim)

loss, recon_loss, latent_loss, latent_loss_scalar, recon_loss_scalar, loss_scalar = vae_loss(data, x_decoded_mean)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.math.maximum(tf.train.exponential_decay(0.001, global_step, 350, 0.9), 0.000002)
lr_scalar = tf.summary.scalar('learning_rate', learning_rate)
merged_training_summary = tf.summary.merge([latent_loss_scalar, recon_loss_scalar, loss_scalar, lr_scalar])

#learning_rate = tf.train.exponential_decay(0.002, global_step, 1000, 0.9)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam_Optimizer').minimize(loss,
                                                                                                global_step=global_step)
init_param = tf.global_variables_initializer()

training_batch_size = 256
n_batches = int(n_train / training_batch_size)
summaryDir = './'
saver = tf.train.Saver()

pretrain = False
modelDir = './model/model.ckpt'
predict = True


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
        sample = sess.run(z_mean, feed_dict={data: X_noise, batch_size: n_train})
        print('sample.shape: ', sample.shape)
        print('sample: ', sample[:3])
        up_assign_op, lambda_assign_op = reinitialzeGMMVariables(sample)
        sess.run([up_assign_op, lambda_assign_op, global_step.assign(0)])
        print('up after reinitial: ', u_p.eval(sess))
        print('lambda_p after reinitial: ', lambda_p.eval(sess))
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
            _, _ce, _rec_loss, _lat_loss, _lr, summary = sess.run([optimizer, loss, recon_loss, latent_loss,learning_rate, merged_training_summary],
                                                                  feed_dict={data: inp, batch_size: training_batch_size})
            if math.isnan(_rec_loss):
                print('recon_loss is NaN')
            if math.isnan(_lat_loss):
                print('latent_loss is NaN')
            progress(j + 1, n_batches, status=' Loss=%f, Lr=%f, Epoch=%d/%d' % (_ce, _lr, i + 1, epoch))
            writer.add_summary(summary, i * n_batches + j)
    save_path = saver.save(sess, "./model/model.ckpt")
    print("Model saved in path: %s" % save_path)
    if predict:
        predictFileName = 'predict_ao.json'
        print('predict results save to: ', predictFileName)
        latent, cluster_prec_temp = sess.run([z, tempGamma], feed_dict={data: X, batch_size: n_train})
        latent = latent.tolist()
        cluster_prec = np.argmax(cluster_prec_temp, axis=1)
        # write latent variable, predict cluster
        # {"index": 0, code": [], "predcit_cluster": 3, "label": 3}
        with open(predictFileName, 'w') as f_write:
            with open('dataset/ccss/ccss_318_414_multi_cuid.json', 'r') as f_read:
                i = 0
                for line in json_lines.reader(f_read):
                    data = {}
                    data['cuid'] = line['cuid']
                    data['ts_game_start_counts_last_30_days_buckets'] = line['ts_game_start_counts_last_30_days_buckets']
                    data['code'] = latent[i]
                    data['predict_cluster'] = cluster_prec[i].item()
                    i += 1
                    json.dump(data, f_write)
                    f_write.write('\n')
