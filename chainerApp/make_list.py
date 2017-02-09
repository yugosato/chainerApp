# -*- coding: utf-8 -*-

import os
import random
import argparse

import chainer


class PreprocessedDataset(chainer.dataset.DatasetMixin):
    
    def __init__(self, path, root, crop_size, random=False):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.crop_size = crop_size
        self.random = random
        

    def __len__(self):
        return len(self.base)
        

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value        
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image /= 255
        return image, label
        
        
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