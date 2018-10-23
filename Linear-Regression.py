import tensorflow as tf
import numpy as np
#1.���峬����
learning_rate=0.01
max_train_step=1000
#2.�������ݼ�
train_x=np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],[9.779],[6.182],[3.41],[4.53],[10.79]],dtype=np.float32)
train_y=np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],[3.366],[2.596],[2.53],[1.221],[2.34]],dtype=np.float32)
total_samples=train_x.shape[0]
#3.����ģ�Ͳ���
X=tf.placeholder(dtype=tf.float32,shape=[None,1]) #None��ʾ֧�� ��������룬1��ʾ����Ϊ1ά
#ģ�Ͳ���
W=tf.Variable(tf.random_normal([1,1]),name="weight")
b=tf.Variable(tf.zeros([1]),name='bias')
#��������ֵ
Y=tf.matmul(X,W)+b

#����ʵ��ֵ
Y_=tf.placeholder(dtype=tf.float32,shape=[None,1])
#������ʧ����
Loss=tf.reduce_sum(tf.pow(Y-Y_,2))/(total_samples)

#�����Ż���
optimizer=tf.train.GradientDescentOptimizer(learning_rate)
#���嵥��ѵ������
train_op=optimizer.minimize(Loss)
log_step=100
#�����Ự
with tf.Session() as sess:
    # ��ʼ��ȫ�ֱ���
    sess.run(tf.global_variables_initializer())
    print("��ʼѵ��")
    for step in range(max_train_step):
        sess.run(train_op,feed_dict={X:train_x,Y_:train_y})
        if step % log_step==0:
           a=sess.run(Loss,feed_dict={X:train_x,Y_:train_y})
           print("step:%d,loss=%.4f,w=%.4f,b=%.4f"%(step,a,sess.run(W),sess.run(b)))
    final_loss=sess.run(Loss,feed_dict={X:train_x,Y_:train_y})
    weight,bias=sess.run([W,b])
import matplotlib.pyplot as plt
plt.plot(train_x,train_y,'ro',label='train_data')
plt.plot(train_x,weight*train_x+bias,label='fitted line')
plt.legend()
plt.show()
