"""
https://github.com/FaBoPlatform/TensorFlow/blob/master/notebooks/HelloWorld.ipynb
"""

import tensorflow as tf


hello = tf.constant("Hello")

sess = tf.Session()
print(sess.run(hello))

print("version:", tf.VERSION)
