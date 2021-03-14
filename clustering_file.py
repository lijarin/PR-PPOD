import os
import sys
import time
import torch
import numpy
import numpy as np

#start_time = time.time()

def pthtovector(file):
    Dict_data = torch.load(file, 'cpu')
    key_value = Dict_data.keys()
    value_list = list(Dict_data.values())

    value_vector = [sum(np.array(value_list[0]).flatten().tolist())/len(np.array(value_list[0]).flatten().tolist())]
    for i in range(1,70):
        value_vector = value_vector + [sum(np.array(value_list[i]).flatten().tolist())/len(np.array(value_list[i]).flatten().tolist())]
    return value_vector

# if __name__ == '__main__':
#     file = '/home/a401/Documents/SSD-COCO/k-means/ssd02.pth'
#     vector = pthtovector(file)
#     print(vector)
    # list_two = []
    # for i in range(len(value_list)):
    #     if i%2 == 0:
    #         value_list_one = np.array(value_list[i].tolist()).flatten().tolist()
    #         list_one.append(value_list_one)
    #     else:
    #         value_list_three = np.array(value_list[i].tolist()).flatten().tolist()
    #         list_two.append(value_list_three)
    # value_final = list_one[0]+list_two[0]+list_one[1]+list_two[1]+list_one[2]+list_two[2]+list_one[3]+list_two[3]+list_one[4]+list_two[4]+list_one[5]+list_two[5]+list_one[6]+list_two[6]+list_one[7]+list_two[7]\
    # +list_one[8]+list_two[8]+list_one[9]+list_two[9]+list_one[10]+list_two[10]+list_one[11]+list_two[11]+list_one[12]+list_two[12]+list_one[13]+list_two[13]+list_one[14]+list_two[14]+list_one[15]+list_two[15]\
    # +list_one[16]+list_two[16]+list_one[17]+list_two[17]+list_one[18]+list_two[18]+list_one[19]+list_two[19]+list_one[20]+list_two[20]+list_one[21]+list_two[21]+list_one[22]+list_two[22]+list_one[23]+list_two[23]\
    # +list_one[24]+list_two[24]+list_one[25]+list_two[25]+list_one[26]+list_two[26]+list_one[27]+list_two[28]+list_one[29]+list_two[29]+list_one[30]+list_two[30]+list_one[31]+list_two[31]+list_one[32]+list_two[32]\
    # +list_one[33]+list_two[33]+list_one[34]+list_two[34]+list_one[35]
#
def filename(path):
    subfile = os.listdir(path)
    subfile.sort()
    return subfile
#
#
def demo_vector():
    path = '/home/a401/Documents/SSD-COCO/k-means'
    modelparameter = filename(path)
    vector = []
    for j in modelparameter:
        p = os.path.join(path,j)
        vector_k_means = pthtovector(p)
        vector.append(vector_k_means)
    return vector
# if __name__ == '__main__':
#     vector = demo_vector()
#     print(vector)
    # b = np.array(a[0][0]).flatten().tolist()
    # for i in range(1,70):
    #     b = b + np.array(a[0][i]).flatten().tolist()
        # b = np.array(a[0][i]).flatten()+np.array(a[1][i]).flatten()+np.array(a[2][i]).flatten()+np.array(a[3][i]).flatten()+np.array(a[4][i]).flatten()+np.array(a[5][i]).flatten()+np.array(a[6][i]).flatten()
    # print(a)

    # print(a[2][70])
    # print(np.array(a[2][70]).flatten())
