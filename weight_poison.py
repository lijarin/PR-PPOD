import os
import sys
import time
import torch
import numpy
import numpy as np

#start_time = time.time()

path_inner = '/home/a401/Documents/SSD-COCO/weight_poison/ssd300_voc2007_110000.pth'
Dict_data = torch.load(path_inner,'cpu')
for key in Dict_data.keys():
    if key == 'vgg.0.weight':
        for i in Dict_data[key]:
            print(i)
        # print(len(Dict_data[key]))

# def demo_vector():
#     path = '/home/a401/Documents/SSD-COCO/weight_poison'
#     modelparameter = filename(path)
#     vector = []
#     for j in modelparameter:
#         p = os.path.join(path,j)
#         vector_k_means = pthtovector(p)
#         vector.append(vector_k_means)
#     return vector
# if __name__ == '__main__':
#     a = demo_vector()
#     print(a)