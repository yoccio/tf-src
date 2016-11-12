# pylint: disable=C0111
# pylint: disable=C0103
# pylint: disable=W0512
#!/User/{env.whoami}}/tensorflow/bin python
#coding:utf-8
import tensorflow as tf
import input_data
mnist = input_data.read_data_sets("tensorflow/src_yoccio/MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
# intialize modul par
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])
# cross-entropy as cost
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000): # 10000 -> 0.9182 1000 -> 0.9137
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# test
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accurancy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accurancy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
