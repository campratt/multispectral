import numpy as np
import matplotlib.pyplot as pl
from spectral import *
import matplotlib as mpl
import spectral.io.envi as envi
import os
from os import listdir


### Read and Calibrate


d = './' # Set to the main directory



def read_image(ID,rwdt=None):

    # Data is too large to be uploaded but can be downloaded at https://hsibraindatabase.iuma.ulpgc.es/

    if rwdt=='raw':
        fn = d+'data/HSI/%s/raw'%ID
        fn_hdr = d+'data/HSI/%s/raw.hdr'%ID
        
    elif rwdt=='white':
        fn = d+'data/HSI/%s/whiteReference'%ID
        fn_hdr = d+'data/HSI/%s/whiteReference.hdr'%ID
        
    elif rwdt=='dark':
        fn = d+'data/HSI/%s/darkReference'%ID
        fn_hdr = d+'data/HSI/%s/darkReference.hdr'%ID
        
    elif rwdt=='truth':
        fn = d+'data/HSI/%s/gtMap'%ID
        fn_hdr = d+'data/HSI/%s/gtMap.hdr'%ID
        
    img = envi.open(fn_hdr, fn)
    arr = img.load()
    
    
    if rwdt!='truth':
        arr = arr[:,:,57:-126]
        
    else:
        pass
    
    
    return arr




def calibrate(ID,step=10):

    # Read data
    r = np.array(read_image(ID,rwdt='raw'))
    d = np.array(read_image(ID,rwdt='dark'))[0]
    w = np.array(read_image(ID,rwdt='white'))[0]

    d2 = d.reshape((1,d.shape[0],d.shape[1]))
    w2 = w.reshape((1,w.shape[0],w.shape[1]))
    
    d2 = d2*np.ones_like(r)
    w2 = w2*np.ones_like(r)

    # Calibrate
    cal = (r - d2)/(w2-d2)
    
    # Smooth by band averaging along the spectral dimension
    inds = np.arange(cal.shape[-1])[::step]

    k = int(step/2)
    inds = inds[inds>k]
    inds = inds[inds<cal.shape[-1]-k-1]    
    cal_new = np.zeros((cal.shape[0],cal.shape[1],len(inds)))
    for i in range(cal.shape[0]):
        for j in range(cal.shape[1]):
            for z in range(len(inds)):
                cal_new[i,j,z] = np.mean(cal[i,j,inds[z]-k:inds[z]+k+1])


    # Normalize
    mins = np.min(cal_new,axis=-1,keepdims=True)
    maxs = np.max(cal_new,axis=-1,keepdims=True)
    cal_new = (cal_new - mins)/(maxs-mins)

    return cal_new
    

    


### Generate cutout images


def random_flip(image):
    flip_x, flip_y = np.random.choice([True, False],size=2)
    if flip_x:
        image = np.flip(image, axis=0)
    if flip_y:
        image = np.flip(image, axis=1)
    return image



def check_box_fit(image, point, pad=8):
    height, width = image.shape
    y, x = point
    
    if x - pad >= 0 and x + pad < width and y - pad >= 0 and y + pad < height:
        return True
    else:
        return False


def generate_images(ID, imagesize=17):

    #np.random.seed(0)

    pad = (imagesize-1)/2

    truth = read_image(ID,rwdt='truth')[:,:,0]
    truth = truth.reshape((truth.shape[0],truth.shape[1]))
    raw = calibrate(ID)


    where_tumor_OG = np.where(truth==2)
    where_normal_OG = np.where(truth==1)


    where_tumor = []
    for w in range(len(where_tumor_OG[0])):
        point = [where_tumor_OG[0][w],where_tumor_OG[1][w]]
        if check_box_fit(truth, point, pad=pad):
            where_tumor.append(point)
    where_tumor = np.array(where_tumor).T

    where_normal = []
    for w in range(len(where_normal_OG[0])):
        point = [where_normal_OG[0][w],where_normal_OG[1][w]]
        if check_box_fit(truth, point, pad=pad):
            where_normal.append(point)
    where_normal = np.array(where_normal).T


    
    min_labels = np.min([where_normal[0].size,where_tumor[0].size])



    

    inds_normal = np.random.choice(np.arange(0,where_normal[0].size),size=min_labels,replace=False)
    inds_tumor = np.random.choice(np.arange(0,where_tumor[0].size),size=min_labels,replace=False)

    for i, ind in enumerate(inds_normal):
        t = int(truth[where_normal[0][i],where_normal[1][i]] - 1)
        truth_classes = np.zeros(2)
        truth_classes[t] = 1
        data_images = raw[int(where_normal[0][ind]-pad):int(where_normal[0][ind]+pad+1),int(where_normal[1][ind]-pad):int(where_normal[1][ind]+pad+1),:]

        fn_x = d +'data/X/%s_%i_%i.npy'%(ID,i,t)
        fn_y = d +'data/Y/%s_%i_%i.npy'%(ID,i,t)
        np.save(fn_x,random_flip(data_images))
        #np.save(fn_x,data_images)
        np.save(fn_y,truth_classes)



    for i, ind in enumerate(inds_tumor):
        t = int(truth[where_tumor[0][i],where_tumor[1][i]] - 1)
        truth_classes = np.zeros(2)
        truth_classes[t] = 1
        data_images = raw[int(where_tumor[0][ind]-pad):int(where_tumor[0][ind]+pad+1),int(where_tumor[1][ind]-pad):int(where_tumor[1][ind]+pad+1),:]
        

        fn_x = d+'data/X/%s_%i_%i.npy'%(ID,i,t)
        fn_y = d+'data/Y/%s_%i_%i.npy'%(ID,i,t)
        
        np.save(fn_x,random_flip(data_images))
        np.save(fn_y,truth_classes)

    



### Run through all images

d_HSI = d + 'data/HSI/'


for ID in sorted(listdir(d_HSI)):
    if '.txt' in ID:
        continue

    t = read_image(ID,rwdt='truth')

    # Only consider images with both normal and tumor tissues
    if 1 in t and 2 in t:
        print('%s has normal/tumor'%ID)
        generate_images(ID,imagesize=17)

    else:
        continue

    

    

