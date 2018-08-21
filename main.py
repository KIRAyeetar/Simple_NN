from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import tensorflow as tf

# =================
#  read iris data
# =================
iris = load_iris()
iris_data = pd.DataFrame(iris.data)
iris_data['target'] = iris.target
iris_data = iris_data.sample(frac=1)

train = iris_data.head(100)
test = iris_data.tail(50)
train['tar_0'] = [1 if i==0 else 0 for i in train['target']]
train['tar_1'] = [1 if i==1 else 0 for i in train['target']]
train['tar_2'] = [1 if i==2 else 0 for i in train['target']]
test['tar_0'] = [1 if i==0 else 0 for i in test['target']]
test['tar_1'] = [1 if i==1 else 0 for i in test['target']]
test['tar_2'] = [1 if i==2 else 0 for i in test['target']]

train_x = np.array(train.loc[:,[0,1,2,3]])
test_x = np.array(test.loc[:,[0,1,2,3]])
train_y = np.array(train.loc[:,['tar_0', 'tar_1', 'tar_2']])
test_y = np.array(test.loc[:,['tar_0', 'tar_1', 'tar_2']])


# =================
#  start to predict
# =================
STEPS = 4000
BATCH_SIZE = 8

x = tf.placeholder(tf.float32, [None, 4])
w = tf.Variable(tf.random_normal([4, 3], stddev=1, seed=1))
b = tf.Variable(tf.random_normal([3], stddev=1, seed=1))

y = tf.nn.softmax(tf.matmul(x, w) + b)
y_ = tf.placeholder(tf.float32, [None, 3])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

count = 0
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(STEPS):
        start = (i*BATCH_SIZE) % len(train_x)
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: train_x[start: end, :],
                                        y_: train_y[start: end, :]})

        # 自己选择是否打印训练时的情况
        # if i % 100 == 0:
        #     print('round '+str(i)+', cross_loss:', sess.run(
        #         cross_entropy, feed_dict={x: train_x[start: end, :],
        #                                   y_: train_y[start: end, :]}))

    # 预测的标签
    # print(sess.run(tf.argmax(y, 1), feed_dict={x: test_x}))
    # 预测的概率
    # print(sess.run(y, feed_dict={x: test_x}))
    # 真实的标签
    # print(sess.run(tf.argmax(test_y, 1)))
    # print(sess.run(w))

    # 打印准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(test_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print('the accuracy is: '+ str(sess.run(accuracy, feed_dict={x: test_x})))
