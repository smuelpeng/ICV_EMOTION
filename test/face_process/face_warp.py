import sys
import os
#import Image
import numpy as np
import cPickle
import cv2
import numpy.linalg as linalg


def crop_image(img, landmarks):
    scale = 1.69
    width,height,channel = img.shape
    scale_w,scale_h = width/scale, height/scale
    h_off,w_off = (height-scale_h)/2, (width-scale_w)/2
    image_scale = img[int(w_off):width-int(w_off),int(h_off):height-int(h_off)]
    return image_scale

def resize_image(img, dsize):
    return cv2.resize(img,dsize)

def alignTransformation(src_pos,dst_pos):
		a=0.0;b=0.0;tx=0.0;ty=0.0;X1=0.0;Y1=0.0;X2=0.0;Y2=0.0;Z=0.0;C1=0.0;C2=0.0;W=2.0
		for i in range(0,2):
				x1=src_pos[i*2]
				y1=src_pos[i*2+1]
				x2=dst_pos[i*2]
				y2=dst_pos[i*2+1]
				Z=Z+x2*x2+y2*y2
				X1=X1+x1
				Y1=Y1+y1
				X2=X2+x2
				Y2=Y2+y2
				C1=C1+x1*x2+y1*y2
				C2=C2+y1*x2-x1*y2
		SolnA=[X2, -Y2, W, 0, Y2, X2, 0, W, Z, 0, X2, Y2, 0, Z, -Y2, X2]
		A=np.array(SolnA,dtype=np.float64).reshape(4,4)
		SolnB=[X1, Y1, C1, C2]
		B=np.array(SolnB,dtype=np.float64).reshape(4,1)
		Soln=np.zeros((4,1),dtype=np.float64)
		cv2.solve(A,B,Soln,cv2.DECOMP_SVD)
		a=Soln[0,0];b=Soln[1,0];tx=Soln[2,0];ty=Soln[3,0]
		norm=a*a+b*b
		a_=a/norm;b_=-b/norm
		tx_=(-a*tx - b*ty)/norm
		ty_=(b*tx - a*ty)/norm
		return a_,b_,tx_,ty_
def calcParameters(src_pos,dst_pos):  #l_x,l_y,r_x,r_y
		[a,b,tx,ty]=alignTransformation(src_pos,dst_pos)
		return a,b,tx,ty	 

REF_SIZE=500
REF_POS=[200,260,287,260,243,332,206,370,281,370] 
SCALAR=4
REF_SIZE = int(REF_SIZE * SCALAR)
REF_POS = [int(REF_POS[i] * SCALAR) for i in range(0,10) ]
def face_warp_main(src_img,landmark_pos):
		eyes_pos=[float(landmark_pos[x]) for x in range(0,10)]
		#print eyes_pos
		src_img_size=src_img.shape
		[a,b,tx,ty]=calcParameters(eyes_pos,REF_POS)
		tranM=np.zeros((2,3),dtype=np.float64)
		tranM[0,0]=a;tranM[1,1]=a
		tranM[0,1]=-b
		tranM[1,0]=b
		tranM[0,2]=tx
		tranM[1,2]=ty
		warpdst=np.zeros((REF_SIZE,REF_SIZE,3),dtype=np.uint8)
		cv2.warpAffine(src_img,tranM,(REF_SIZE,REF_SIZE),warpdst)
		tranM_new=np.zeros((2,3),dtype=np.float64)
		new_landmark_pos=[]
		for idx in range(0,5):
				old_pos_x=float(landmark_pos[idx*2])
				old_pos_y=float(landmark_pos[idx*2+1])
				new_pos_x=tranM[0,0]*old_pos_x+tranM[0,1]*old_pos_y+tranM[0,2]
				new_pos_y=tranM[1,0]*old_pos_x+tranM[1,1]*old_pos_y+tranM[1,2]
				new_landmark_pos.append(new_pos_x)
				new_landmark_pos.append(new_pos_y)
		return warpdst,new_landmark_pos
