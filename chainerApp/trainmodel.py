# -*- coding: utf-8 -*-

import numpy as np
import argparse
import os
import time
import datetime
import chainer
from chainer import training
from chainer.training import extensions
from dataset import PreprocessedDataset
from makefigure import logplot

import mymodel
import nin
import alex
import lenet5
import vgg16
import googlenet
import deepface


def train():    
    archs = {
        'mymodel': mymodel.MyModel,
        'nin': nin.NIN,
        'alex': alex.Alex,
        'lenet': lenet5.Lenet5,
        'vgg': vgg16.VGG16,
        'googlenet': googlenet.GoogLeNet,
        'deepface': deepface.DeepFace
    }    
    
    parser = argparse.ArgumentParser(description='Training convnet from dataset (only 3 channels image)')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('test', help='Path to test image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='nin',
                        help='Convnet architecture')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Training minibatch size')
    parser.add_argument('--test_batchsize', '-b', type=int, default=250,
                        help='Test minibatch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')  
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    args = parser.parse_args()    
   
    print 'GPU: {}'.format(args.gpu)
    print '# Minibatch-size: {}'.format(args.batchsize)
    print '# epoch: {}'.format(args.epoch)
    print ''    

    # Initialize model to train
    model = archs[args.arch]()
    
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        
    # Load datasets
    train = PreprocessedDataset(args.train, args.root, model.insize)
    test = PreprocessedDataset(args.test, args.root, model.insize)
        
    # Set up iterator
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.test_batchsize, repeat=False, shuffle=False)
        
    # Set up optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
        
    # Set up trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)
    
    # Copy chain with shared parameters to flip 'train' flag only in test
    eval_model = model.copy()
    eval_model.train = False
         
    trainer.extend(extensions.Evaluator(test_iter, eval_model, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport([
        'epoch', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    
    # Get date and time
    date = datetime.datetime.today()
    
    start_time = time.clock()
    trainer.run() 
    total_time = datetime.timedelta(seconds = time.clock() - start_time)
        
    # Save trained model
    print ''
    print 'Training has been finished.'
    print 'Total training time: {}.'.format(total_time)
    print 'Saving the trained model...',
    chainer.serializers.save_npz(os.path.join(args.out, 'model_final_' + args.arch), model)
    print '----> done'
        
    logplot(args.out)
    info = open(os.path.join(args.out, 'info'), 'a')
    info.write('Date: {}.\n'.format(date.strftime("%Y/%m/%d %H:%M:%S")))
    info.write('----> Total training time: {}.'.format(total_time))
    

if __name__ == '__main__':
    train()
    