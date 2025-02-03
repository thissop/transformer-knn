#    Edited by Sizhuo Li
#    Author: Ankit Kariryaa, University of Bremen
#    Edited by: Jesse Meyer, NASA

from keras.ops import where, sum, power
from jax import jit

newaxis = None

@jit
def tversky(y_true, y_pred, alpha=0.6, smooth=1):
    y_t = where(y_true == 1,  1.0, 0.0)
    y_w = where(y_true == 10, 2.0, 1.0)

    tp = sum(y_pred * y_t)
    fp = alpha * sum((y_pred * y_w) * (1 - y_t))
    fn = (1.0 - alpha) * sum(((1 - y_pred) * y_w) * y_t)

    numerator = tp
    denominator = tp + fp + fn
    
    score = (numerator + smooth) / (denominator + smooth)
    return 1 - score


#https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
@jit
def focal_tversky(y_true, y_pred, alpha=0.7):
    loss = tversky(y_true, y_pred, alpha)
    return power(loss, 0.75)


@jit
def accuracy(y_true, y_pred):
    y_t = where(y_true == 1, 1.0, 0.0)

    tp = sum(y_pred * y_t)
    fp = sum(y_pred * (1 - y_t))
    fn = sum((1 - y_pred) * y_t)
    tn = sum((1 - y_pred) * (1 - y_t))
    return (tp + tn) / (tp + tn + fp + fn)


@jit
def dice_coef(y_true, y_pred, smooth=0.0000001):
    y_t = where(y_true == 1, 1.0, 0.0)

    tp = sum(y_pred * y_t)
    fp = sum(y_pred * (1 - y_t))
    fn = sum((1 - y_pred) * y_t)

    two_tp = 2.0 * tp
    return (two_tp + smooth) / (two_tp + fp + fn + smooth)

