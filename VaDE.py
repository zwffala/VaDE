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

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


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
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w


def load_data(dataset):
    path = 'dataset/'+dataset+'/'
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
        X = np.concatenate((x_train,x_test))
        Y = np.concatenate((y_train,y_test))
        
    if dataset == 'reuters10k':
        data=scio.loadmat(path+'reuters10k.mat')
        X = data['X']
        Y = data['Y'].squeeze()
        
    if dataset == 'har':
        data=scio.loadmat(path+'HAR.mat')
        X=data['X']
        X=X.astype('float32')
        Y=data['Y']-1
        X=X[:10200]
        Y=Y[:10200]

    return X,Y

def config_init(dataset):
    if dataset == 'mnist':
        return 784,3000,10,0.002,0.002,10,0.9,0.9,1,'sigmoid'
    if dataset == 'reuters10k':
        return 2000,15,4,0.002,0.002,5,0.5,0.5,1,'linear'
    if dataset == 'har':
        return 561,120,6,0.002,0.00002,10,0.9,0.9,5,'linear'
        
def gmmpara_init():
    
    theta_init=np.ones(n_centroid)/n_centroid
    u_init=np.zeros((latent_dim,n_centroid))
    lambda_init=np.ones((latent_dim,n_centroid))

    theta_p = tf.Variable(theta_init, dtype=tf.float32, name="pi")
    u_p = tf.Variable(u_init, dtype=tf.float32, name="u")
    lambda_p = tf.Variable(lambda_init, dtype=tf.float32, name="lambda")

    return theta_p,u_p,lambda_p


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

    temp_p_c_z = tf.exp(tf.reduce_sum(tf.log(temp_theta_tensor3)-0.5*tf.log(2*math.pi*temp_lambda_tensor3)
                                      -tf.square(temp_Z-temp_u_tensor3)/(2*temp_lambda_tensor3), axis=1))+1e-10
    return temp_p_c_z/tf.reduce_sum(temp_p_c_z, axis=-1, keepdims=True)

# def get_gamma(tempz):
#     temp_Z=T.transpose(K.repeat(tempz,n_centroid),[0,2,1])
#     temp_u_tensor3=T.repeat(u_p.dimshuffle('x',0,1),batch_size,axis=0)
#     temp_lambda_tensor3=T.repeat(lambda_p.dimshuffle('x',0,1),batch_size,axis=0)
#     temp_theta_tensor3=theta_p.dimshuffle('x','x',0)*T.ones((batch_size,latent_dim,n_centroid))
#
#     temp_p_c_z=K.exp(K.sum((K.log(temp_theta_tensor3)-0.5*K.log(2*math.pi*temp_lambda_tensor3)-\
#                        K.square(temp_Z-temp_u_tensor3)/(2*temp_lambda_tensor3)),axis=1))+1e-10
#     return temp_p_c_z/K.sum(temp_p_c_z,axis=-1,keepdims=True)


def vae_loss(x, x_decoded_mean):
    Z = tf.expand_dims(z, -1)        # z.shape(batch_size, latent)
    Z = tf.tile(Z, [1, 1, n_centroid])  # Z.shape(batch_size, latent, K)

    Z_mean_t = tf.expand_dims(z_mean, -1)        # z_mean.shape(batch_size, latent)
    Z_mean_t = tf.tile(Z_mean_t, [1, 1, n_centroid])

    Z_log_var_t = tf.expand_dims(z_log_var, -1)
    Z_log_var_t = tf.tile(Z_log_var_t, [1, 1, n_centroid]) # Z_log_var_t.shape(batch, latent, K)

    u_tensor3 = tf.expand_dims(u_p, 0)  # u_tensor3.shape(latent, K)
    u_tensor3 = tf.tile(u_tensor3, [batch_size, 1, 1])  # u_tensor3.shape(batch_size, latent, K)

    lambda_tensor3 = tf.expand_dims(lambda_p, 0)
    lambda_tensor3 = tf.tile(lambda_tensor3, [batch_size, 1, 1])

    theta_tensor3 = tf.expand_dims(theta_p, 0)
    theta_tensor3 = tf.tile(theta_tensor3, [latent_dim, 1])
    theta_tensor3 = tf.expand_dims(theta_tensor3, 0)
    theta_tensor3 = tf.tile(theta_tensor3, [batch_size, 1, 1])

    p_c_z = tf.exp(tf.reduce_sum(tf.log(theta_tensor3)-0.5*tf.log(2*math.pi*lambda_tensor3)-tf.square(Z-u_tensor3)/(2*lambda_tensor3), axis=1))+1e-10   # p_c_z.shape(batch, K)
    gamma = p_c_z/tf.reduce_sum(p_c_z, axis=-1, keepdims=True)
    gamma_t = tf.expand_dims(gamma, axis=1)
    gamma_t = tf.tile(gamma_t, [1, latent_dim, 1])  # gamma_t.shape(batch, latent, 1)

    latent_loss = tf.reduce_sum(0.5*gamma_t*(latent_dim*tf.log(math.pi*2)+tf.log(lambda_tensor3)
                                             +tf.exp(Z_log_var_t)/lambda_tensor3+tf.square(Z_mean_t-u_tensor3)/lambda_tensor3), axis=(1,2)) \
                  -0.5*tf.reduce_sum(z_log_var+1, axis=-1) \
                  -tf.reduce_sum(tf.log(tf.tile(tf.expand_dims(theta_p, 0), [batch_size, 1]))*gamma, axis=-1) \
                  +tf.reduce_sum(tf.log(gamma)*gamma, axis=-1)   # latent_loss.shape(batch, )
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

#
# def lr_decay():
#     if dataset == 'mnist':
#         adam_nn.lr.set_value(floatX(max(adam_nn.lr.get_value()*decay_nn,0.0002)))
#         adam_gmm.lr.set_value(floatX(max(adam_gmm.lr.get_value()*decay_gmm,0.0002)))
#     else:
#         adam_nn.lr.set_value(floatX(adam_nn.lr.get_value()*decay_nn))
#         adam_gmm.lr.set_value(floatX(adam_gmm.lr.get_value()*decay_gmm))
#     print ('lr_nn:%f'%adam_nn.lr.get_value())
#     print ('lr_gmm:%f'%adam_gmm.lr.get_value())
    
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
A `Tensor` which represents input layer of a model. Its shape
    is (batch_size, first_layer_dimension) and its dtype is `float32`.
    first_layer_dimension is determined based on given `feature_columns`.
'''


def resolve_variational_autoencoder(data, original_dim, intermediate_dim, latent_dim, datatype, trainable=True):
    # construct the encoder dense layers
    # _input = tf.placeholder(shape=(None, original_dim), name='input', dtype=tf.float32)
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

    gamma = get_gamma(z)

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
                             activation=None)  # activation=(tf.nn.sigmoid if variational else None))
    decoder_layers.append(_layer)
    result = _layer.apply(result)

    return _input, encoder_layers, z_mu, z_sigma, z, gamma, decoder_layers, result


dataset = 'mnist'
db = sys.argv[1]
if db in ['mnist','reuters10k','har']:
    dataset = db
print ('training on: ' + dataset)
ispretrain = False
# batch_size = 100
batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
latent_dim = 10
intermediate_dim = [500,500,2000]
accuracy=[]
X,Y = load_data(dataset)
original_dim,epoch,n_centroid,lr_nn,lr_gmm,decay_n,decay_nn,decay_gmm,alpha,datatype = config_init(dataset)
data = tf.placeholder(tf.float32, shape=(None, original_dim), name='data')
label = tf.placeholder(tf.float32, shape=(None, original_dim), name='label')
theta_p,u_p,lambda_p = gmmpara_init()

x, encoder_layers, z_mean, z_log_var, z, tempGamma, decoder_layers, x_decoded_mean = resolve_variational_autoencoder(data, original_dim, intermediate_dim, latent_dim, datatype)

loss, latent_loss_scalar, recon_loss_scalar, loss_scalar = vae_loss(x, x_decoded_mean)
merged_summary_op = tf.summary.merge_all()

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.math.maximum(tf.train.exponential_decay(lr_nn, global_step, 7000, 0.9), 0.0002)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam_Optimizer').minimize(loss, global_step=global_step)
init_param = tf.global_variables_initializer()

n_train = 70000
training_batch_size = 100
n_batches = int(n_train / training_batch_size)
modelDir = './'

with tf.Session() as sess:
    sess.run(init_param)
    aidx = list(range(n_train))
    writer = tf.summary.FileWriter(modelDir, sess.graph)
    for i in range(epoch):
        random.shuffle(aidx)
        ptr = 0
        for j in range(n_batches):
            inp = X[aidx[ptr:ptr + training_batch_size], :]
            ptr += training_batch_size
            _, _ce, _lr, summary = sess.run([optimizer, loss, learning_rate, merged_summary_op], feed_dict={data: inp, label: inp, batch_size: training_batch_size})

            # acc = cluster_acc(tf.math.argmax(tempGamma, axis=1), Y[aidx[ptr:ptr + batch_size]])
            # acc = cluster_acc(np.argmax(cluster_prec, axis=1), Y[aidx[ptr:ptr + batch_size]])
            progress(j + 1, n_batches, status=' Loss=%f, Lr=%f, Epoch=%d/%d' % (_ce, _lr, i + 1, epoch))
            writer.add_summary(summary, i*n_batches+j)
        cluster_prec = sess.run(tempGamma, feed_dict={data: X, label: X, batch_size: n_train})
        acc = cluster_acc(np.argmax(cluster_prec, axis=1), Y)
        print('acc_p_c_z: ', acc[0])
        acc_tensor = tf.constant(acc[0], name='acc_p_c_z')
        summary = tf.summary.scalar('acc_p_c_z', acc_tensor)
        acc_summary = sess.run(summary)
        writer.add_summary(acc_summary)
