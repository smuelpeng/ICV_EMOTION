import numpy as np
import cv2
import cPickle
import sys
import time
import copy
from face_warp import *
from multiprocessing import Pool
poch_size=100
def write_landmark(img_list):
    for idx,line in enumerate(img_list):
        filename = line.split()[0]
        landmark =[int(float(la)) for la in line.split()[1:5]]
        img = cv2.imread(filename)
	landmarks=line.split()[1:11]
	imgs,lands = face_warp_main(img,landmarks)
	img_crop = crop_image(imgs,lands)	
        newname='FACE_ALIGN/'+filename.split('/')[-1]
        cv2.imwrite(newname,img_crop)
if __name__=="__main__":
    image_lists=[line.strip() for line in open(sys.argv[1])]
    image_lists_length=len(image_lists)
    task_lists=[image_lists[i*poch_size:(i+1)*poch_size] for i in range(image_lists_length/poch_size)]
    if  not image_lists_length%poch_size==0:
        task_lists.append(image_lists[(image_lists_length/poch_size)*poch_size:])
    pool=Pool(40)
    pool.map(write_landmark,tuple(task_lists))
    pool.close()
    pool.join()
    
