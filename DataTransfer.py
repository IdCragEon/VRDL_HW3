#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import os
import pandas as pd
import h5py


train_folder = "D:\Program\98semester\VRDL\HW3\\train"
test_folder = "D:\Program\98semester\VRDL\HW3\\test"
extra_folder = "D:\Program\98semester\VRDL\HW3\\extra"
resize_size = (64,64)

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
    return attrs

def img_boundingbox_data_constructor(mat_file):
    f = h5py.File(mat_file,'r') 
    all_rows = []
    print('image bounding box data construction starting...')
    bbox_df = pd.DataFrame([],columns=['height','img_name','label','left','top','width'])
    for j in range(f['/digitStruct/bbox'].shape[0]):
        img_name = get_name(j, f)
        row_dict = get_bbox(j, f)
        row_dict['img_name'] = img_name
        all_rows.append(row_dict)
        bbox_df = pd.concat([bbox_df,pd.DataFrame.from_dict(row_dict,orient = 'columns')])
    bbox_df['bottom'] = bbox_df['top']+bbox_df['height']
    bbox_df['right'] = bbox_df['left']+bbox_df['width']
    print('finished image bounding box data construction...')
    return bbox_df

img_folder = train_folder
mat_file_name = 'digitStruct.mat'
h5_name = 'train_data_processed.h5'
img_bbox_data = img_boundingbox_data_constructor(os.path.join(img_folder,mat_file_name))

def collapse_col(row):
    global resize_size
    global train_folder
    new_row = {}
    new_row['index'] = int(list(row['img_name'])[0].split('.')[0])
    new_row['img_name'] = train_folder + '\\' + list(row['img_name'])[0]
    new_row['labels'] = row['label'].astype(np.str).str.cat(sep='_')
    new_row['top'] = row['top'].astype(np.str).str.cat(sep='_')
    new_row['left'] = row['left'].astype(np.str).str.cat(sep='_')
    new_row['bottom'] = row['bottom'].astype(np.str).str.cat(sep='_')
    new_row['right'] = row['right'].astype(np.str).str.cat(sep='_')
    new_row['width'] = row['width'].astype(np.str).str.cat(sep='_')
    new_row['height'] = row['height'].astype(np.str).str.cat(sep='_')
    new_row['num_digits'] = len(row['label'].values)
    return pd.Series(new_row,index=None)

img_bbox_data_grouped = img_bbox_data.groupby('img_name').apply(collapse_col)
img_bbox_data_grouped.index.name = None

img_bbox_data_grouped.sort_values(by=['index'],inplace=True)
img_bbox_data_grouped.set_index('index',inplace=True)

fp = open("traindata.txt", "w")
for index,row in img_bbox_data_grouped.iterrows():
    img_name = row['img_name']
    label = [int(i) for i in[float(j) for j in row['labels'].split('_')]]
    y_min = [int(i) for i in[float(j) for j in row['top'].split('_')]]  
    x_min = [int(i) for i in[float(j) for j in row['left'].split('_')]]
    y_max = [int(i) for i in[float(j) for j in row['bottom'].split('_')]]
    x_max = [int(i) for i in[float(j) for j in row['right'].split('_')]]
    num = row['num_digits']
    bbox = []
    for i in range(num):
        bbox.append(str(x_min[i])+','+str(y_min[i])+','+str(x_max[i])+','+str(y_max[i])+','+str(label[i]))
    bbox_str = ' '.join(bbox)
    fp.write(img_name+' '+bbox_str+'\n')

fp.close()




