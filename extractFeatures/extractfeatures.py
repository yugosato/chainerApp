# -*- coding: utf-8 -*-

import argparse
import os
import sys
import cPickle as pickle
import numpy as np
from chainer import cuda
from dataset import PreprocessedDataset


xp = cuda.cupy

def main():    
    parser = argparse.ArgumentParser(description='Extract features from trained convnet model (only 3 channels image)')
    parser.add_argument('test', help='Path to test image-label list file')
    parser.add_argument('trainedmodel', help='Trainedmodel for extacting features')                
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')  
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    args = parser.parse_args()    
    
    try:
        os.makedirs(args.out)
    except OSError:
        pass   

    # Initialize the model to train
    model = pickle.load(open(os.path.join('trainedmodel', args.trainedmodel), 'rb'))
    model.to_gpu()
        
    # Load the datasets
    test = PreprocessedDataset(args.test, args.root, 224)
             
    # Extract features and write dat file
    imagelist = np.genfromtxt(args.test, dtype=np.str)[:,0]
    N = len(imagelist)
  
    results = []
    for i in range(len(test)):
        print imagelist[i]
        sys.stderr.write('{} / {}\r'.format(i, N))
        sys.stderr.flush()
        image, label = test.get_example(i)
        x = image.reshape((1,) + image.shape)
        y = model(inputs={'data': xp.asarray(x)}, outputs=['fc7'], train=False)
        outputs = cuda.to_cpu(y[0].data)
        results.append(outputs[0])
    np.save(os.path.join(args.out, 'features.npy'), results)

    
if __name__ == '__main__':
    main()
    