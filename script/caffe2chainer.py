# -*- coding: utf-8 -*-

from chainer.functions import caffe
import cPickle as pickle
import argparse
import os


def extractFilename(path):
    base = os.path.basename(path)
    fname, ext = os.path.splitext(base)
    return fname

    
def main():    
    parser = argparse.ArgumentParser(description='Convert caffemodel to chainermodel')
    parser.add_argument('model', help='Path to caffemodel')
    parser.add_argument('--out', '-o', default='chainermodel', help='Output directory')
    args = parser.parse_args()    

    try:
        os.makedirs(args.out)
    except OSError:
        pass    
    
    caffemodel = caffe.CaffeFunction(args.model)
    modelname = extractFilename(args.model)
    pickle.dump(caffemodel, open(os.path.join(args.out, modelname + '.pkl'), 'wb'), -1)

    
if __name__ == '__main__':
    main()