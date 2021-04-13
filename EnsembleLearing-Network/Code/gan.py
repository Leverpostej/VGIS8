import numpy as np
import os
import tensorflow as tf
import h5py
from sklearn.utils import shuffle
import scipy.io as sio

from Code.models import *
from Code.loss import *

class GAN:
    def __init__(self, w, h, c, lr, m_s, b_s, restore_epoch, con_train, direc):
        #Image dimensions
        self.w = w
        self.h = h
        self.c = c     
        #Training parameters
        self.base_lr = lr
        self.max_step = m_s
        self.batch_size = b_s
        self.restore_epoch = restore_epoch
        self.con_tr = con_train
        self.direc = direc      
    
    def load_vgg(self, sess):
        vgg_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg')
        
        print('Initialize VGG network ... ')
        weights = np.load('VGG/vgg19.npy', encoding='latin1', allow_pickle=True).item()
        layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                  'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                  'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                  'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']

        for p, q in enumerate(layers):
            print(p, q, weights[q][0].shape, weights[q][1].shape)
            sess.run(vgg_params[2*p].assign(weights[q][0]))
            sess.run(vgg_params[2*p+1].assign(weights[q][1]))
    
    def setup(self):
        self.real_lr = tf.placeholder(tf.float32, 
                                     [None, self.w, self.h, self.c],
                                     name='input_lr')
        self.real_hr = tf.placeholder(tf.float32, 
                                     [None, self.w, self.h, self.c],
                                     name='input_hr')
        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
        self.forward()
    
    def forward(self):
        
        with tf.variable_scope('GAN') as scope:
            self.fake_hr = generator(self.real_lr, scope='g')
            self.d_real_hr = discriminator(self.real_hr, scope='d')           
            
            scope.reuse_variables()
                               
            self.d_fake_hr = discriminator(self.fake_hr, scope='d')
            
            scope.reuse_variables()      
            
            alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)

            # gradient penalty
            interpolates = alpha*tf.reshape(self.real_hr, [self.batch_size, -1])+(1-alpha)*tf.reshape(self.fake_hr, [self.batch_size, -1])
            interpolates = tf.reshape(interpolates, [self.batch_size, self.w, self.h, self.c])
            gradients = tf.gradients(discriminator(interpolates, scope='d'), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis = 1))
            self.gradient_penalty = tf.reduce_mean((slopes-1.)**2)
                
        with tf.variable_scope('vgg') as scope:
            self.vgg_real_hr = vgg_model(self.real_hr)
            scope.reuse_variables()
            self.vgg_fake_hr = vgg_model(self.fake_hr)

    def loss(self):
        
        lambda_a = 0.01
        lambda_b = 0.01
        lambda_c = 1.
        lambda_d = 10.
        
        #adversarial loss, selecting WGAN
        self.adv_loss = generator_loss(type = 'wgan', fake = self.d_fake_hr)
        
        #gradient loss
        self.gra_loss_hr = gradient_loss(self.real_hr, self.fake_hr)
        
        #MSE
        self.mse_hr = mse_loss(self.real_hr, self.fake_hr)
        
        #Perceptual loss
        self.per_loss_hr = per_loss(self.vgg_real_hr, self.vgg_fake_hr)
                
        #GAN Generator loss
        self.g_loss = self.adv_loss + lambda_a * self.gra_loss_hr + lambda_b * self.mse_hr + lambda_c * self.per_loss_hr
        
        #GAN Discriminator loss
        self.w_dis = discriminator_loss(type = 'wgan', real = self.d_real_hr, fake = self.d_fake_hr)
        
        # improved WGAN Discriminator loss
        self.d_loss = self.w_dis + lambda_d * self.gradient_penalty
        
        #Isolate variables
        self.vars = tf.trainable_variables()
        d_vars = [v for v in self.vars if 'd' in v.name]
        g_vars = [v for v in self.vars if 'g' in v.name]
        
        #Train while freezing other variables
        optimizer = tf.train.AdamOptimizer(self.lr)
        
        self.d_train = optimizer.minimize(self.d_loss, var_list=d_vars)
        self.g_train = optimizer.minimize(self.g_loss, var_list=g_vars)
        
    def train(self):
        self.setup()
        self.loss()

        init = (tf.global_variables_initializer(), 
                tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep = None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            sess.run(init)
            self.load_vgg(sess)
                                                        
            for epoch in range(self.restore_epoch, self.max_step):
                for k in range(6): # split dataset into 6 parts, the first five parts contains 1000*49 images
                    if k < 5:
                        data = np.zeros((49000, 80, 80, 1))
                        label = np.zeros((49000, 80, 80, 1)) 
                    else:
                        data = np.zeros((744*49, 80, 80, 1))
                        label = np.zeros((744*49, 80, 80, 1))
                    print('Begin loading data')
                    path_full='LR/LR.h5'               #load LR images
                    f = h5py.File(path_full,'r')
                    load_data = f['data']
                    data[:,:,:,0] = load_data
                    print('Finish loading data!')
                    
                    print('Begin loading label')
                    path_full='HR/HR.h5'               #load HR images
                    f = h5py.File(path_full)
                    load_data = f['data']
                    label[:,:,:,0] = load_data
                    print('Finish loading label!')
                    #==========================================================================
                    data, label = shuffle(data, label)
                        
                    num_batches = data.shape[0] // self.batch_size
                    
                    print('Start training part %d' % (k+1))
                    
                    if epoch < 50:
                        current_lr = self.base_lr
                    else:
                        current_lr = self.base_lr * (100 - epoch) / 50
    
                    for i in range(num_batches):
                        batch_data = data[i*self.batch_size : (i+1)*self.batch_size]
                        batch_label = label[i*self.batch_size : (i+1)*self.batch_size]
                                                
                        # Train G
                        run_list = [self.g_train, self.g_loss, self.adv_loss, self.mse_hr, self.gra_loss_hr, self.per_loss_hr]
                        feed_dict = {self.real_lr: batch_data, self.real_hr: batch_label, self.lr: current_lr}                       
                        _, temp_loss_g, tmp_adv, tmp_mse, tmp_gra, tmp_per = sess.run(run_list, feed_dict)
                        
                        for kk in range(4):
                            #Optimize D
                            run_list = [self.d_train, self.d_loss, self.w_dis]
                            feed_dict = {self.real_lr: batch_data, self.real_hr: batch_label, self.lr: current_lr}
                            _, temp_loss_d, tmp_wdis = sess.run(run_list, feed_dict)
                                                
                        print('[Epoch: %d, Part: %d, Batch: %d], G: %.4f, D: %.4f' % (epoch, k, i, temp_loss_g, temp_loss_d))
                                                
                if not os.path.exists('./Network/' + self.direc):
                    os.makedirs('./Network/' + self.direc)
                if (epoch + 1) % 2 == 0:              
                    saver.save(sess, './Network/'+ self.direc + '/' + repr(epoch + 1) + '.ckpt')
            
    def test(self):
        self.setup()
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(), 
                tf.local_variables_initializer())
        data = np.zeros((3000, 320, 320, 1))
        recon_hr = np.zeros((3000, 320, 320))
                
        print('Begin loading label')
        path_full='HR/HR.h5'               #load HR images
        f = h5py.File(path_full)
        load_data = f['data'][0:3000,:,:]
        data[:,:,:,0] = load_data
        print('Finish loading label!')
        
        num_batches = data.shape[0] // self.batch_size
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, './Network/'+ self.direc + '/' + repr(self.restore_epoch) + '.ckpt')
                    
            for i in range(num_batches):
                if i < num_batches:
                    batch_data = data[i*self.batch_size : (i+1)*self.batch_size]
                else:
                    batch_data = data[i*self.batch_size: 3000]
     
                run_list = self.fake_hr
                feed_dict = {self.real_lr: batch_data}

                tmp_fake_hr = sess.run(run_list, feed_dict)
                
                tmp_fake_hr = np.asarray(tmp_fake_hr)
                
                if i < num_batches:
                    tmp_fake_hr = tmp_fake_hr.reshape(self.batch_size, 320, 320)
                    recon_hr[i*self.batch_size : (i+1)*self.batch_size] = tmp_fake_hr
                else:
                    tmp_fake_hr = tmp_fake_hr.reshape(3000-self.batch_size*i, 320, 320)
                    recon_hr[i*self.batch_size : 3000] = tmp_fake_hr
                    
            if not os.path.exists('./Result/' + self.direc):
                os.makedirs('./Result/' + self.direc)
            path = 'Result/' + self.direc + '/' + repr(self.restore_epoch) + '.hdf5'
            f = h5py.File(path, 'w')
            f.create_dataset('recon_hr', data=recon_hr)
            f.close()    