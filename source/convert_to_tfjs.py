import tensorflowjs as tfjs
import tensorflow as tf

if __name__ == '__main__':
    model = tf.keras.models.load_model('model/result/fp128_128.h5')
    tfjs.converters.save_keras_model(model, 'model/result/model_tfjs/')
