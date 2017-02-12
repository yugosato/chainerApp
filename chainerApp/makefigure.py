# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import json
import os


def logplot(out, logname='log'):    
    try:
        os.makedirs(out)
    except OSError:
        pass    
        
    jsondata = open(os.path.join(out, logname), 'r')
    jsondata = json.load(jsondata)
    
    epoch = []
    loss = []
    acc = []   
 
    for data in jsondata:
        epoch.append(data['epoch'])
        loss.append(data['validation/main/loss'])
        acc.append(data['validation/main/accuracy'])
    
    plt.figure(1)
    plt.plot(epoch, loss, '-b')
    plt.xlim(0, len(epoch))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.savefig(os.path.join(out, 'loss.png'))
    
    plt.figure(2)
    plt.xlim(0, len(epoch))
    plt.plot(epoch, acc, '-r')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.savefig(os.path.join(out, 'accuracy.png'))