## add pictures
#!/usr/bin/python -tt
from PIL import ImageTk, Image
import os, numpy, PIL
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import timeit
os.chdir("C:\\Python27\\6_5eV\\14V\\3800ns")
os.getcwd()
allfiles=os.listdir(os.getcwd())
imlist=[filename for filename in allfiles if filename[-4:] in [".bmp", ".BMP"]]
w, h=(1280, 960)
N=len(imlist)
arr=np.zeros((h, w), np.float)
for im in imlist:
	imarr=numpy.array(Image.open(im), dtype=numpy.float)
	arr=arr+imarr/N
arr=numpy.array(numpy.round(arr), dtype=numpy.uint8)
cm=plt.get_cmap('jet')
colored_image=cm(arr)
Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save('test.png')

## colorbar and plot with counts
img=Image.open("A:\\Maria\\Expenses\\MARIA\\Fe(CO)5\\Test Folder\\Test_Fe(CO)3_Denoised.png")
pixels=list(img.getdata())
width,height=img.size
pixels=[pixels[i*width:(i+1)*width] for i in xrange(height)]
a=np.array(pixels)
arry=a.sum(0)
plt.show()
plt.plot(arry)
plt.show()
plt.subplot(1,1,1)
plt.imshow(img, cmap='jet')
plt.clim(0, 245090)
plt.colorbar(extend='both')
plt.show()

## add colorbar
imarr=numpy.array(Image.open("A:\\Maria\\Expenses\\MARIA\\Fe(CO)5\\0.2 eV - Fe(CO)4\\test.png"), dtype=numpy.float)
arr=numpy.array(numpy.round(imarr), dtype=numpy.uint8)
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(arr)
plt.colorbar(m)
plt.imshow(arr)
plt.show()

## or with the colorbar from -1 to 1
plt.subplot(1,1,1)
plt.imshow(img, cmap='jet')
plt.clim(-1,1)
plt.colorbar(extend='both')
plt.show()

## colorplot with counts
colors=img.convert('RGB').getcolors()
for i in range (1, 253):
	x,y=colors[i]
	a.append(x)
	b.append(y)
	m.append(i)
C=np.array(b)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(m, a, c=C/255.0)
plt.show()

## denoise picture
from os import listdir
from os.path import isfile, join
import cv2
mypath = 'A:\\home\\Desktop\\Expenses\\MARIA\\Test Folder'
onlyfiles = [img for img in listdir(mypath) if isfile(join(mypath, img))]
i=0
for img in onlyfiles:
	img = cv2.imread(img)
	b,g,r = cv2.split(img)
	rgb_img = cv2.merge([r,g,b])
	dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
	b,g,r = cv2.split(dst)
	rgb_dst = cv2.merge([r,g,b])
	plt.imsave('pic' + str(i) +'.jpg', rgb_dst, cmap='jet')
	i+=1

##kinetic energies
#!/usr/bin/python -tt
from PIL import ImageTk, Image
import os, numpy, PIL
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import timeit
import cv2
from skimage import io, filters
from scipy import ndimage
from skimage import measure
import imutils
import argparse
ap=argparse.ArgumentParser()
args=vars(ap.parse_args())
image=cv2.imread("A:\\Maria\\Expenses\\Python\\Pictures\\result_bw.png")
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred=cv2.GaussianBlur(gray, (5,5), 0)
thresh=cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
cnts=cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts=cnts[0] if imutils.is_cv2() else cnts[1]
for c in cnts:
	M=cv2.moments(c)
	cX=int(M["m10"] / M["m00"])
	cY=int(M["m01"] / M["m00"])
	cv2.drawContours(img, [c], -1, (0,255,0),2)
	cv2.circle(img, (cX, cY), 7, (0,0,0), -1)
center=(cX,cY)

img=cv2.imread("C:\\Python27\\Picture.png")
d=[]
m=[]
radius=40
cX,cY=(653,466)
os.chdir("A:\\Maria\\Expenses\\Python\\Pictures")
os.getcwd()
h,w=img.shape[:2]
mask=np.zeros((h,w), np.uint8)
for i in range(1, 960):
	for j in range(1, 1280):
		if ((i-cY)**2+(j-cX)**2<=z**2 and (i-cY)**2+(j-cX)**2>=(z-1)**2):
				d.append([i,j])
for z in range(1, radius/2):
	cv2.circle(mask, (cX,cY), 2*z, 255, -1)
	res=cv2.bitwise_and(img, img, mask=mask)
	cv2.imwrite('pic'+str(z)+'.png', res)
	img1=cv2.imread('pic'+str(z)+'.png')
	cv2.circle(mask, (cX,cY), 2*(z-1), 0, -1)
	res=cv2.bitwise_and(img1, img1, mask=mask)
	cv2.imwrite('picz'+str(z)+'.png', res)
m=[]
for z in range(1, radius/2):
	im=io.imread('picz'+str(z)+'.png', as_gray=True)
	val=filters.threshold_otsu(im)
	drops=ndimage.binary_fill_holes(im>val)
	labels=measure.label(drops)
	m.append(labels.max())
len(m)
k=len(m)
l=[]
l=m[:][::-1]
l
plt.plot(l)