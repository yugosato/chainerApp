# -*- coding: utf-8 -*-

import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class VGG16(Chain):
    '''
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    Karen Simonyan, Andrew Zisserman
    ICLR2015
    '''
    insize = 224

    
    def __init__(self):
        super(VGG16, self).__init__(
            conv1_1=L.Convolution2D(None, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(None, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(None, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(None, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(None, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(None, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(None, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(None, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(None, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(None, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(None, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(None, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(None, 512, 3, stride=1, pad=1),

            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 4096),
            fc8=L.Linear(None, 1000)
        )
        self.train = True

        
    def __call__(self, x, t):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), train=self.train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train, ratio=0.5)
        h = self.fc8(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss