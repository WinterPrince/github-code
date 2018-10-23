import tensorflow as tf
import numpy as np
#1.定义超参数
learning_rate=0.01
max_train_step=1000
#2.输入数据集
train_x=np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],[9.779],[6.182],[3.41],[4.53],[10.79]],dtype=np.float32)
train_y=np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],[3.366],[2.596],[2.53],[1.221],[2.34]],dtype=np.float32)
total_samples=train_x.shape[0]
#3.定义模型参数
X=tf.placeholder(dtype=tf.float32,shape=[None,1]) #None表示支持 任意个输入，1表示数据为1维
#模型参数
W=tf.Variable(tf.random_normal([1,1]),name="weight")
b=tf.Variable(tf.zeros([1]),name='bias')
#定义推理值
Y=tf.matmul(X,W)+b

#定义实际值
Y_=tf.placeholder(dtype=tf.float32,shape=[None,1])
#定义损失函数
Loss=tf.reduce_sum(tf.pow(Y-Y_,2))/(total_samples)

#创建优化器
optimizer=tf.train.GradientDescentOptimizer(learning_rate)
#定义单步训练操作
train_op=optimizer.minimize(Loss)
log_step=100
#创建会话
with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())
    print("开始训练")
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
