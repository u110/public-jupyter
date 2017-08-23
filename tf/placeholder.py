"""
https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/placeholder.ipynb
"""
import tensorflow as tf

"""
定数
"""
const_a = tf.constant([1, 2, 3, 4, 5, 6], shape=(3,2))
print(const_a)

sess = tf.Session()
result_a = sess.run(const_a)
print("----")
print("result_a", result_a)


variable_a = tf.Variable([
    [1.0, 1.0],
    [2.0, 2.0],
])
print("variable_a", variable_a)


"""
変数
"""
sess = tf.Session()
# initialize
sess.run(variable_a.initializer)

result_a = sess.run(variable_a)
print("----")
print(result_a)


"""
Placeholder
"""

ph_a = tf.placeholder(tf.int16)
ph_b = tf.placeholder(tf.int16)

add_op = tf.add(ph_a, ph_b)

sess = tf.Session()
result_a = sess.run(add_op, feed_dict={ph_a:2, ph_b:3})
print("Placeholder", result_a)
