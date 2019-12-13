import tensorflow as tf
from keras import backend as K


def aa_acc(prots_oh, reconstructed):
    x, fx = K.argmax(prots_oh, axis=-1), K.argmax(reconstructed, axis=-1)
    non_dash_mask = tf.greater(x, 0)
    aa_acc = K.sum(tf.cast(tf.boolean_mask(K.equal(x, fx), non_dash_mask), 'float32'))
    aa_acc /= K.sum(tf.cast(non_dash_mask, 'float32'))
    return aa_acc