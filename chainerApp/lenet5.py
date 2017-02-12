# -*- coding: utf-8 -*-

import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class Lenet5(Chain):
    '''
    GradientBased Learning Applied to Document Recognition
    Yann LeCun, Leon Bottou, Yoshua Bengjo, Patrick Haffner
    Proc. of the IEEE 1998
    '''
    insize = 152
    
    def __init__(self):
        super(Lenet5, self).__init__(
            conv1=L.Convolution2D(None, 6, 5),
            conv2=L.Convolution2D(None, 16, 5),
            conv3=L.Convolution2D(None, 120, 5),
            fc4=L.Linear(None, 84),
            fc5=L.Linear(None, 10)
        )
        self.train = True

        
    def __call__(self, x, t):
        h = F.sigmoid(F.average_pooling_2d(self.conv1(x), 2))
        h = F.sigmoid(F.average_pooling_2d(self.conv2(h),2))
        h = self.conv3(h)
        h = F.tanh(self.fc4(h))
        h = self.fc5(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss