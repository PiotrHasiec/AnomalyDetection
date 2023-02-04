import tensorflow as tf

def n(number):
    if number<10:
        return '0'+str(number)
    return str(number)



def root_percentage_mean_error(y_true, y_pred):
    err = tf.sqrt( tf.keras.metrics.mean_squared_error(y_true,y_pred)/tf.reduce_mean(tf.square(y_pred)))
    return err
