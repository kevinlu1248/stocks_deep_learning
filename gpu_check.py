import tensorflow as tf

tf.autograph.set_verbosity(
    10
)
print(tf.config.list_physical_devices('GPU'))
print(tf.config.experimental.list_physical_devices('GPU'))
print(tf.test.is_gpu_available())
