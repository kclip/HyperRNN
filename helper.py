import numpy as np
import tensorflow as tf
from scipy.special import binom
from tensorflow.contrib.layers import xavier_initializer
import math 
def init_weights(shape,name):
    return tf.get_variable(str(name)+'_w', shape, tf.float32, xavier_initializer())
def init_bias(shape,name):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals,name=name+'_b')    
def normal_full_layer(input_layer, size,name):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size],name)
    b = init_bias([size],name)
    return tf.matmul(input_layer, W) + b
def normal_full_layer2(input_size, size,name):
    W = init_weights([input_size, size],name)
    b = init_bias([size],name)
    return W,b
def normal_full_layer2_nobias(input_size, size,name):
    W = init_weights([input_size, size],name)
    return W
def full_layer_no_bias(input_layer, size,name):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size],name)
    return tf.matmul(input_layer, W) 
def normal_full_layer_withWandB(input_layer, W,b):
    return tf.matmul(input_layer, W) + b
def standart_gaussian_noise_layer(shape):
    noise = tf.random_normal(shape=shape) 
    return noise

def frange(x, y, jump):
    while x <= y:
        yield x
        x += jump


        
        
def M_MIMO_DNN_CE(X,output_number,N0,batch_size,M,K,B,L,Lp,alpha_para):

    X_tilde_ini= tf.Variable(tf.sqrt(1/M)*standart_gaussian_noise_layer([M, 2*L]),trainable=True,name='x')
    power_normal=tf.sqrt(tf.reduce_sum(tf.square(X_tilde_ini[:,0:L])+tf.square(X_tilde_ini[:,L:2*L]),axis=0))
    X_tilde=X_tilde_ini/(tf.concat([power_normal,power_normal],axis=0))
    
    
    #X_tilde_ini= tf.Variable(tf.sqrt(1/M)*standart_gaussian_noise_layer([M, 2*L]),trainable=True,name='x')
    #power_normal=tf.reduce_sum(tf.square(X_tilde_ini))
    #X_tilde=X_tilde_ini/tf.sqrt(power_normal/L)#loose power normalization
    X_tilde_complex=tf.complex(X_tilde[:,0:L],X_tilde[:,L:2*L])
    y=tf.matmul(X, X_tilde_complex) 
    y_real=tf.concat([tf.real(y),tf.imag(y)],axis=1)
    noise=tf.sqrt(N0/2)*standart_gaussian_noise_layer((batch_size,L*2))
    y_noise=y_real+noise
    
    u1 =(normal_full_layer(y_noise,1024,'u1'))
    n_u1 = tf.nn.relu(tf.layers.batch_normalization(u1,name='u1_normalization'))
    u2 =(normal_full_layer(n_u1,512,'u2'))
    n_u2 = tf.nn.relu(tf.layers.batch_normalization(u2,name='u2_normalization'))
    u3 =(normal_full_layer(n_u2,256,'u3'))
    n_u3 = tf.nn.relu(tf.layers.batch_normalization(u3,name='u3_normalization'))
    #q =tf.sign(2*tf.nn.sigmoid(alpha_para*normal_full_layer(n_u3,B,'u4'))-1.0)
    #q =2*tf.nn.sigmoid(alpha_para*normal_full_layer(n_u3,B,'u4'))-1.0
    q =tf.nn.tanh(alpha_para*normal_full_layer(n_u3,B,'u4'))
    #print(q)
    r1 =(normal_full_layer(q,1024,'r1'))
    n_r1 = tf.nn.relu(tf.layers.batch_normalization(r1,name='r1_normalization'))
    r2 =(normal_full_layer(n_r1,512,'r2'))
    n_r2 = tf.nn.relu(tf.layers.batch_normalization(r2,name='r2_normalization'))
    r3 =(normal_full_layer(n_r2,512,'r3'))
    n_r3 = tf.nn.relu(tf.layers.batch_normalization(r3,name='r3_normalization'))
    h_est =normal_full_layer(n_r3,M*2,'r4')
    return h_est,X_tilde
def M_MIMO_DNN_CE_hyper_RNN_downlink(X1,X2,S,N0,batch_size,M,B,L1,L2,Lp):


    
    ###################HYPER###################
    X_tilde_ini2= tf.Variable(tf.sqrt(1/1)*standart_gaussian_noise_layer([1, 2*L2]),trainable=True,name='x2')
    power_normal2=tf.sqrt(tf.reduce_sum(tf.square(X_tilde_ini2[:,0:L2])+tf.square(X_tilde_ini2[:,L2:2*L2]),axis=0))
    X_tilde2=X_tilde_ini2/(tf.concat([power_normal2,power_normal2],axis=0))
    X_tilde_complex2=tf.reshape(tf.complex(X_tilde2[:,0:L2],X_tilde2[:,L2:2*L2]),[1,1,L2]) # 1,L2
    
    X_tilde_complex2_full=tf.tile(X_tilde_complex2,[batch_size,1,1]) #(batch_size,1,L2)
    #X2(batch_size *S *M) X2[:,1,:]:(batch_size *M) 
    y2=tf.matmul(tf.reshape(X2[:,0,:],[batch_size,M,1]), X_tilde_complex2_full) # (batch_size ,M,1) (batch_size,1,L2)  =(batch_size *M *L2) 
    y22=tf.reshape(y2,[batch_size,1,M *L2])
    for s in range(S-1):
        temp=tf.matmul(tf.reshape(X2[:,s+1,:],[batch_size,M,1]), X_tilde_complex2_full) #(batch_size *M *L2) 
        temp2=tf.reshape(temp,[batch_size,1,M *L2])
        y22=tf.concat([y22,temp2],axis=1)
    y_real2=tf.concat([tf.real(y22),tf.imag(y22)],axis=2)
    noise2=tf.sqrt(N0/2)*standart_gaussian_noise_layer((batch_size,S,M*L2*2))
    y_noise2=y_real2+noise2#(batch_size *S *2L2) 
    
    
    U1,B1 =normal_full_layer2(2*L2*M,1024,'U')
    W1 =normal_full_layer2_nobias(1024,1024,'H')
    V1,C1 =normal_full_layer2(1024,B+256+256,'V')
    H0=tf.nn.relu(tf.matmul(y_noise2[:,0,:],U1)+B1) #batch_size ,1024
    hyper_y=tf.reshape(tf.matmul(H0,V1)+C1,[batch_size,B+256+256,1])#batch_size ,B+512+256+128
    for s in range(S-1):
        H0=tf.nn.relu(tf.matmul(y_noise2[:,s+1,:],U1)+B1+tf.matmul(H0,W1))
        y_temp=tf.reshape(tf.matmul(H0,V1)+C1,[batch_size,B+256+256,1])# U V W
        hyper_y=tf.concat([hyper_y,y_temp],axis=2)
        
    #print(hyper_y)    
    ###################HYPER###################
    
    X_tilde_ini= tf.Variable(tf.sqrt(1/M)*standart_gaussian_noise_layer([M, 2*L1]),trainable=True,name='x1')
    power_normal=tf.sqrt(tf.reduce_sum(tf.square(X_tilde_ini[:,0:L1])+tf.square(X_tilde_ini[:,L1:2*L1]),axis=0))
    X_tilde=X_tilde_ini/(tf.concat([power_normal,power_normal],axis=0))
    X_tilde_complex=tf.reshape(tf.complex(X_tilde[:,0:L1],X_tilde[:,L1:2*L1]),[1,M,L1])# M,L1
    
    X_tilde_complex1_full=tf.tile(X_tilde_complex,[batch_size,1,1]) #(batch_size,M,L1)   
    y=tf.matmul(X1, X_tilde_complex1_full) # (batch_size,S,M) (batch_size,M,L1) =(batch_size *S *L1) 
    y_real=tf.concat([tf.real(y),tf.imag(y)],axis=2)
    noise=tf.sqrt(N0/2)*standart_gaussian_noise_layer((batch_size,S,L1*2))
    y_noise=y_real+noise# (batch_size,S,2*L1)
    
   
    
    
    
    u1,ub1 =normal_full_layer2(2*L1,256,'r1')
    u2,ub2 =normal_full_layer2(256,256,'r2')
    u3,ub3 =normal_full_layer2(256,128,'r3')
    u4,ub4 =normal_full_layer2(128,B,'r4')
    
    nu1=tf.nn.relu(tf.matmul(y_noise[:,0,:],u1)+ub1)
    nu2=tf.nn.relu(tf.matmul(nu1,u2)+ub2)
    nu3=tf.nn.relu(tf.matmul(nu2,u3)+ub3)
    q=tf.nn.tanh(tf.matmul(nu3,u4)+ub4)
    #print(q)
    #print(hyper_y[:,0:B,0])
    U2,B2 =normal_full_layer2(B,256,'U2')
    W2 =normal_full_layer2_nobias(256,256,'H2')
    V2,C2 =normal_full_layer2(256,M*2,'V2')
    H2=tf.nn.relu(tf.matmul(q*hyper_y[:,0:B,0],U2)+B2) #batch_size ,1024
    #print(H2)
    h_est=tf.reshape(tf.matmul(H2*hyper_y[:,B:B+256,0],V2)+C2,[batch_size,1,M*2])#batch_size ,B+512+256+128
    for s in range(S-1):
        nu1=tf.nn.relu(tf.matmul(y_noise[:,s+1,:],u1)+ub1)
        nu2=tf.nn.relu(tf.matmul(nu1,u2)+ub2)
        nu3=tf.nn.relu(tf.matmul(nu2,u3)+ub3)
        q=tf.nn.tanh(tf.matmul(nu3,u4)+ub4)
        H2=tf.nn.relu(tf.matmul(q*hyper_y[:,0:B,s+1],U2)+B2+tf.matmul(H2,W2)*hyper_y[:,B+256:B+256+256,s+1])
        y_temp=tf.reshape(tf.matmul(H2*hyper_y[:,B:B+256,s+1],V2)+C2,[batch_size,1,M*2])
        h_est=tf.concat([h_est,y_temp],axis=1)
        
    #print(h_est)
    h_est_complext=tf.complex(h_est[:,:,0:M],h_est[:,:,M:2*M])
    return h_est_complext


def M_MIMO_DNN_CE_hyper_downlinktime(X1_1,X1_2,X2,S,N0,batch_size,M,B,L1,L2,Lp):


    
    ###################HYPER###################
    X_tilde_ini2= tf.Variable(tf.sqrt(1/M)*standart_gaussian_noise_layer([M, 2*L2]),trainable=True,name='x2')
    power_normal2=tf.sqrt(tf.reduce_sum(tf.square(X_tilde_ini2[:,0:L2])+tf.square(X_tilde_ini2[:,L2:2*L2]),axis=0))
    X_tilde2=X_tilde_ini2/(tf.concat([power_normal2,power_normal2],axis=0))
    X_tilde_complex2=tf.reshape(tf.complex(X_tilde2[:,0:L2],X_tilde2[:,L2:2*L2]),[1,M,L2]) # M,L2
    
    X_tilde_complex2_full=tf.tile(X_tilde_complex2,[batch_size,1,1]) #(batch_size,M,L2)   
    y2=tf.reshape(tf.matmul(X2, X_tilde_complex2_full),[batch_size,S*L2]) # (batch_size *S *M) (batch_size,M,L2)  
    y_real2=tf.concat([tf.real(y2),tf.imag(y2)],axis=1)
    noise2=tf.sqrt(N0/2)*standart_gaussian_noise_layer((batch_size,S*L2*2))
    y_noise2=y_real2+noise2
    
    h1 =(normal_full_layer(y_noise2,1024,'h1'))
    n_h1 = tf.nn.relu(tf.layers.batch_normalization(h1,name='h1_normalization'))
    h2 =(normal_full_layer(n_h1,1024,'h2'))
    n_h2 = tf.nn.relu(tf.layers.batch_normalization(h2,name='h2_normalization'))
    h3 =(normal_full_layer(n_h2,1024,'h3'))
    n_h3 = tf.nn.relu(tf.layers.batch_normalization(h3,name='h3_normalization'))
    hyper =normal_full_layer(n_h3,B+2*M+512+256+128,'h4')
    
    ###################HYPER###################
    
    X_tilde_ini= tf.Variable(tf.sqrt(1/M)*standart_gaussian_noise_layer([M, 2*L1]),trainable=True,name='x1')
    power_normal=tf.sqrt(tf.reduce_sum(tf.square(X_tilde_ini[:,0:L1])+tf.square(X_tilde_ini[:,L1:2*L1]),axis=0))
    X_tilde=X_tilde_ini/(tf.concat([power_normal,power_normal],axis=0))
    X_tilde_complex=tf.complex(X_tilde[:,0:L1],X_tilde[:,L1:2*L1])# M,L1
    
    y=tf.matmul(X1_1, X_tilde_complex) # (batch_size,M) (M,L1)
    y_real=tf.concat([tf.real(y),tf.imag(y)],axis=1)
    noise=tf.sqrt(N0/2)*standart_gaussian_noise_layer((batch_size,L1*2))
    y_noise=y_real+noise# (batch_size,L1)
    
    u1 =(normal_full_layer(y_noise,512,'u1'))
    n_u1 = tf.nn.relu(tf.layers.batch_normalization(u1,name='u1_normalization'))
    u2 =(normal_full_layer(n_u1,256,'u2'))
    n_u2 = tf.nn.relu(tf.layers.batch_normalization(u2,name='u2_normalization'))
    u3 =(normal_full_layer(n_u2,128,'u3'))
    n_u3 = tf.nn.relu(tf.layers.batch_normalization(u3,name='u3_normalization'))
    #q =tf.sign(2*tf.nn.sigmoid(alpha_para*normal_full_layer(n_u3,B,'u4'))-1.0)
    #q =2*tf.nn.sigmoid(alpha_para*normal_full_layer(n_u3,B,'u4'))-1.0
    q =tf.nn.tanh(normal_full_layer(n_u3,B,'u4'))
    #print(q)
    new_q=tf.concat([q,tf.real(X1_2),tf.imag(X1_2)],axis=1)
    #print(new_q)
    temp_q=new_q*hyper[:,0:B+2*M]
    r1 =(normal_full_layer(temp_q,512,'r1'))
    n_r1 = tf.nn.relu(tf.layers.batch_normalization(r1,name='r1_normalization'))
    temp_r1=n_r1*hyper[:,B+2*M:B+2*M+512]
    r2 =(normal_full_layer(temp_r1,256,'r2'))
    n_r2 = tf.nn.relu(tf.layers.batch_normalization(r2,name='r2_normalization'))
    temp_r2=n_r2*hyper[:,B+2*M+512:B+2*M+512+256]
    r3 =(normal_full_layer(temp_r2,128,'r3'))
    n_r3 = tf.nn.relu(tf.layers.batch_normalization(r3,name='r3_normalization'))
    temp_r3=n_r3*hyper[:,B+2*M+512+256:B+2*M+512+256+128]
    h_est =normal_full_layer(temp_r3,M*2,'r4')
    h_est_complext=tf.complex(h_est[:,0:M],h_est[:,M:2*M])
    
    return h_est_complext

