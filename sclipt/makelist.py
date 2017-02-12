# -*- coding: utf-8 -*-

import os
import random
import argparse
import chainer


class DatasetListFromDirectory(chainer.dataset.DatasetMixin):
    
    def __init__(self, root='.'):
        directories = os.listdir(root)
        label_table = []
        pairs = []
        for dir_index, directory in enumerate(directories):
            full_dir_path = os.path.join(root, directory)
            if not os.path.isdir(full_dir_path): continue
           
            label_table.append((dir_index, directory))
            
            for file_name in os.listdir(full_dir_path):
                input_path = os.path.join(full_dir_path, file_name)
                print input_path
                if not os.path.isfile(input_path): continue
                            
                pairs.append((input_path, dir_index))
   
                
        self._pairs = pairs
        self._label_table = label_table
        
        self.create_label_list()
        self.create_image_list()
        

    def create_label_list(self):
        with open('label_list.txt', 'w') as f:
            for (label_index, label_name) in self._label_table:
                f.write('{} {}\n'.format(label_index, label_name))
                
    
    def create_image_list(self):
        with open('image_list.txt', 'w') as f:
            for (input_path, label_index) in self._pairs:
                f.write('{} {}\n'.format(input_path, label_index))
       
                
def main():    
    parser = argparse.ArgumentParser(description='Make label and image list')
    parser.add_argument('root', help='Root directory path of image files')
    args = parser.parse_args()
    
    DatasetListFromDirectory(args.root)
    
    
if __name__ == '__main__':    
    main()