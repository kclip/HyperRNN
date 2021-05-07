import numpy as np
import tensorflow as tf
from scipy.special import binom
from scipy.io import savemat
import helper as hp
import datetime
import math 

            
def M_MIMO_DNN_CE(ID=1,M=64,B=6,Lp=2,S=10,L=2,L2=2,delta=1,SNRdb=10,batch_size=1024,mini_batch=1,total_epoch=20000):
    #M transmit annennas
    #K=1 number of users
    #B bits for each user;
    #L pilot length
    #Lp number of paths
    #print('M_MIMO_DNN_CE:'+str(ID)+', transmit annennas:'+str(M)+', Users:'+str(K)+\
    #', B(feedback bandwith):'+str(B)+', pilots:'+str(L)+\
    #', Paths:'+str(Lp)+', SNR(train-test):'+str(SNRdb))
   
    np.random.seed(0)
    K=1
    file_name='Com_M_MIMO_DNN_CE(M='+str(M)+')K('+str(K)+')Q('+str(B)+')L1('+str(L)+')Lp('+str(Lp)+')SNRdb('+str(SNRdb)+')'
    print(file_name)
    input_number=M
    output_number=M*2
    tf.reset_default_graph()
    X = tf.placeholder("complex64", [batch_size, input_number])
    y_true = tf.placeholder("float32", [batch_size, output_number])
    alpha_para = tf.placeholder("float32", [])
    N0_dnn = tf.placeholder("float32", [])
    y_pred,test1=hp.M_MIMO_DNN_CE(X,output_number,N0_dnn,batch_size,M,K,B,L,Lp,alpha_para) # B* M1
    #cross = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred),1))
    cross = tf.reduce_mean(tf.pow(y_true - y_pred, 2))*2.0
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    start_time = datetime.datetime.now()
    interval=(datetime.datetime.now()-start_time).seconds
    time_index=1
    avg_cost = 0.
    cost_array=np.zeros((int(total_epoch/10),1),dtype=float)
    cost_index=0
    with tf.Session() as sess:
    # Training         
        sess.run(init)
        for epoch in range(total_epoch):
            
            if epoch<total_epoch/3:
                learning_R=0.001
            elif epoch<(3.0*total_epoch/4):
                learning_R=0.0001
            else:
                learning_R=0.00001
            SNR = 10**(SNRdb/10.0)
            sigma2=1/SNR
            #if epoch==0:
            #    alpha_para_input=0.5
            #else:
            #    alpha_para_input=alpha_para_input*1.01
                #if alpha_para_input<10:
                #    alpha_para_input=10.0
            alpha_para_input=1.0
            for index_m in range(mini_batch): 
                theta =60.0*np.random.rand(batch_size,K,Lp)-30.0
                alpha=np.sqrt(0.5)*(np.random.standard_normal([batch_size,K,Lp])+1j*np.random.standard_normal([batch_size,K,Lp]))
                h=np.zeros([batch_size,M*2],dtype=float)
                x_h=np.zeros([batch_size,M],dtype=complex)
                for p in range(Lp):
                    for m in range(M):
                        temp1=np.exp(1j*2*np.pi*0.5*m*np.sin(theta[:,0,p]/180*np.pi))
                        temp2=1.0/np.sqrt(Lp)*alpha[:,0,p]*temp1
                        x_h[:,m]=x_h[:,m]+temp2
                h[:,0:M]=np.real(x_h)
                h[:,M:2*M]=np.imag(x_h)
                #print(h[0,:])
                #print(h[1,:])

                N0_input=sigma2



                _,cs,test11=sess.run([optimizer,cross,test1], feed_dict={X:x_h,y_true:h,N0_dnn:N0_input,alpha_para:alpha_para_input,learning_rate:learning_R})
                avg_cost += cs 
            #print(cs)
            #print(a2)
            #print(input_index)
            #print(sm_index_batch)
            if (epoch+1) % 10 == 0:
                #print(test11)
                print("Epoch:",'%04d' % (epoch+1), "train_cost=", \
                "{:.9f}".format(avg_cost/10))
                
                cost_array[cost_index,0]=avg_cost/10
                cost_index +=1
                avg_cost=0.
            if (epoch+1) % 1000 == 0:                 
                #train_val= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                train_val= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                a=sess.run(train_val)
                train_val2= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                #print(train_val)

                #print((train_val[0].name[0:-2]))
                store_dic={}
                store_dic['cost_array']=cost_array
                for zz in range(len(train_val)):
                    dd=train_val[zz].name[0:-2]
                    dd = dd.replace("_normalization/", "_")
                    #np.savetxt(file_name+dd+'.dat', a[zz])
                    store_dic[dd]=np.array(a[zz])
                    if train_val2[-1].name == train_val[zz].name:
                        break
                savemat(file_name+".mat",store_dic)
       
    
    

def M_MIMO_DNN_CE_hyper_RNN_downlink2(ID=1,M=64,B=6,Lp=2,S=10,L1=2,L2=2,time_doppler_opt=1,SNRdb=10,batch_size=1024,mini_batch=1,total_epoch=20000):
    #M transmit annennas
    #K=1 number of users
    #B bits for each user;
    #L pilot length
    #Lp number of paths
    #print('M_MIMO_DNN_CE:'+str(ID)+', transmit annennas:'+str(M)+', Users:'+str(K)+\
    #', B(feedback bandwith):'+str(B)+', pilots:'+str(L)+\
    #', Paths:'+str(Lp)+', SNR(train-test):'+str(SNRdb))
   
    np.random.seed(0)
    delta=100
    file_name='hyper_Com_M_MIMO_DNN_CE_RNN_downlink_fixed_correlation(M='+str(M)+')K('+str(1)+')Q('+str(B)+')S('+str(S)+')L1('+str(L1)+')L2('+str(L2)+')delta('+str(delta)+')opt('+str(time_doppler_opt)+')Lp('+str(Lp)+')SNRdb('+str(SNRdb)+')'
    print(file_name)
    tf.reset_default_graph()
    X1 = tf.placeholder("complex64", [batch_size,S, M])
    X2 = tf.placeholder("complex64", [batch_size,S, M])
    y_true = tf.placeholder("complex64", [batch_size,S, M])
    delta_number=0.05*(delta*1e6+3e9)/3e8
    N0_dnn = tf.placeholder("float32", [])
    y_pred=hp.M_MIMO_DNN_CE_hyper_RNN_downlink(X1,X2,S,N0_dnn,batch_size,M,B,L1,L2,Lp) # B* M1
    
    #cross = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred),1))
    cross = tf.reduce_mean(tf.square(tf.abs(y_true - y_pred)))
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    start_time = datetime.datetime.now()
    interval=(datetime.datetime.now()-start_time).seconds
    time_index=1
    avg_cost = 0.
    cost_array=np.zeros((int(total_epoch/10),1),dtype=float)
    if time_doppler_opt==1:#0.5e-4
        dl_cor=0.9998
        ul_cor=0.9998
    if time_doppler_opt==2:#0.5e-3
        dl_cor=0.9829
        ul_cor=0.9818
    if time_doppler_opt==3:#0.8e-3
        dl_cor=0.9566
        ul_cor=0.9537
    if time_doppler_opt==4:#1e-3
        dl_cor=0.9326
        ul_cor=0.9281
    if time_doppler_opt==5:#0.2e-2
        dl_cor=0.7441
        ul_cor=0.7280
    if time_doppler_opt==6:#0.3e-2
        dl_cor=0.4720
        ul_cor=0.4422
    if time_doppler_opt==7:#0.4e-2
        dl_cor=0.1698
        ul_cor=0.1304
    cost_index=0
    with tf.Session() as sess:
    # Training         
        sess.run(init)
        for epoch in range(total_epoch):
            
            if epoch<total_epoch/3:
                learning_R=0.001
            elif epoch<(3.0*total_epoch/4):
                learning_R=0.0001
            else:
                learning_R=0.00001
            SNR = 10**(SNRdb/10.0)
            sigma2=1/SNR
            for index_m in range(mini_batch): 
                theta =60.0*np.random.rand(batch_size,Lp)-30.0
                alphaX=np.sqrt(0.5)*(np.random.standard_normal([batch_size,Lp])+1j*np.random.standard_normal([batch_size,Lp]))
                alphaY=np.sqrt(0.5)*(np.random.standard_normal([batch_size,Lp])+1j*np.random.standard_normal([batch_size,Lp]))
                alpha2=np.sqrt(0.5)*(np.random.standard_normal([batch_size,S,Lp])+1j*np.random.standard_normal([batch_size,S,Lp]))
                alpha1=np.sqrt(0.5)*(np.random.standard_normal([batch_size,S,Lp])+1j*np.random.standard_normal([batch_size,S,Lp]))
                alpha1[:,0,:]=alphaX
                alpha2[:,0,:]=alphaY
                for s in range(S):
                    if s>0:
                        noise=np.sqrt(0.5)*(np.random.standard_normal([batch_size,Lp])+1j*np.random.standard_normal([batch_size,Lp]))
                        alpha1[:,s,:]=alpha1[:,s-1,:]*dl_cor+noise*np.sqrt(1-np.square(dl_cor)) #downlink
                        noise=np.sqrt(0.5)*(np.random.standard_normal([batch_size,Lp])+1j*np.random.standard_normal([batch_size,Lp]))
                        alpha2[:,s,:]=alpha2[:,s-1,:]*ul_cor+noise*np.sqrt(1-np.square(ul_cor)) #uplink
                h_dl=np.zeros([batch_size,S,M],dtype=complex)
                h_ul=np.zeros([batch_size,S,M],dtype=complex)
                for p in range(Lp):
                    for m in range(M):
                        for s in range(S):
                            temp1=np.exp(1j*2*np.pi*delta_number*m*np.sin(theta[:,p]/180*np.pi))
                            temp2=1.0/np.sqrt(Lp)*alpha2[:,s,p]*temp1
                            h_ul[:,s,m]=h_ul[:,s,m]+temp2
                            
                            temp1=np.exp(1j*2*np.pi*0.5*m*np.sin(theta[:,p]/180*np.pi))
                            temp2=1.0/np.sqrt(Lp)*alpha1[:,s,p]*temp1
                            h_dl[:,s,m]=h_dl[:,s,m]+temp2
                #print(h_ul)
                #print(h_dl)

                N0_input=sigma2



                _,cs=sess.run([optimizer,cross], feed_dict={X1:h_dl,X2:h_ul,y_true:h_dl,N0_dnn:N0_input,learning_rate:learning_R})
                avg_cost += cs 
            #print(cs)
            #print(a2)
            #print(input_index)
            #print(sm_index_batch)
            if (epoch+1) % 10 == 0:
                #print(test11)
                #print("Epoch:",'%04d' % (epoch+1), "train_cost=", \
                #"{:.9f}".format(avg_cost/10))
                
                cost_array[cost_index,0]=avg_cost/10
                cost_index +=1
                avg_cost=0.
            if (epoch+1) % 1000 == 0:                 
                #train_val= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                train_val= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                a=sess.run(train_val)
                train_val2= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                #print(train_val)

                #print((train_val[0].name[0:-2]))
                store_dic={}
                store_dic['cost_array']=cost_array
                for zz in range(len(train_val)):
                    dd=train_val[zz].name[0:-2]
                    dd = dd.replace("_normalization/", "_")
                    #np.savetxt(file_name+dd+'.dat', a[zz])
                    store_dic[dd]=np.array(a[zz])
                    if train_val2[-1].name == train_val[zz].name:
                        break
                savemat(file_name+".mat",store_dic)
                
                
