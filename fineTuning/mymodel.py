# -*- coding: utf-8 -*-

import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class MyModel(Chain):
    insize = 224
    
    def __init__(self, model):
        super(MyModel, self).__init__(
            base=model,
            fc8=L.Linear(None, 5748)
        )
        self.train = True

        
    def __call__(self, x, t):
        h = self.base(inputs={'data': x}, outputs=['fc7'], train=self.train)
        h = self.fc8(h[0])

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss