# -*- coding: utf-8 -*-

import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class DeepFace(Chain):
    '''
    DeepFace: Closing the Gap to Human-Level Performance in Face Verification
    Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf
    CVPR2014
    '''
    insize = 152
    
    def __init__(self):
        super(DeepFace, self).__init__(
            conv1=L.Convolution2D(None, 32, 11),               
            conv2=L.Convolution2D(None, 16, 9),
            conv3=L.Convolution2D(None, 16, 9),
            conv4=L.Convolution2D(None, 16, 7),
            conv5=L.Convolution2D(None, 16, 5),
            fc6=L.Linear(None, 4096),            
            fc7=L.Linear(None, 5749)
        )
        self.train = True

        
    def __call__(self, x, t):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))        
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = self.fc7(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss