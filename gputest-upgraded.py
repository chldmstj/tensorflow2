import tensorflow as tf

# Creates a graph.
#tf.debugging.set_log_device_placement(True)
with tf.device('/device:GPU:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# Runs the op.
#tf.debugging.set_log_device_placement(True)
print(sess.run(c))


#tf.config.set_soft_device_placement(True)
#tf.debugging.set_log_device_placement(True)

# try:
#   # 유효하지 않은 GPU 장치를 명시
#   with tf.device('/device:XLA_GPU:0'):
#     a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#     b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#     c = tf.matmul(a, b)
#     print(c)
# except RuntimeError as e:
#   print(e)


# strategy = tf.distribute.MirroredStrategy()
#
# with strategy.scope():
#   inputs = tf.keras.layers.Input(shape=(1,))
#   predictions = tf.keras.layers.Dense(1)(inputs)
#   model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
#   model.compile(loss='mse',
#                 optimizer=tf.keras.optimizers.SGD(learning_rate=0.2))
#
# gpus = tf.config.experimental.list_logical_devices('GPU')
# if gpus:
#   # 여러 GPU에 계산을 복제
#   c = []
#   for gpu in gpus:
#     with tf.device(gpu.name):
#       a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#       b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#       c.append(tf.matmul(a, b))
#
#   with tf.device('/device:GPU:0'):
#     matmul_sum = tf.add_n(c)
#
#   print(matmul_sum)
