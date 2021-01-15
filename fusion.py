import pywt
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread('lena1.jpg',0) #read gray image.
# img1 = cv.resize(img1, (256,256))
img2 = cv.imread('lena2.jpg',0) #read gray image.
# img2 = cv.resize(img2, (256,256))

coeffs2 = pywt.swt2(img1, 'db1',level=1)
coeffs2 = coeffs2[0]
LL, (LH, HL, HH) = coeffs2

A1L1 = np.float32(LL)
H1L1 = np.float32(LH)
V1L1 = np.float32(HL)
D1L1 = np.float32(HH)

coeffs2 = pywt.swt2(img2, 'db1',level=1)
coeffs2 = coeffs2[0]
LL, (LH, HL, HH) = coeffs2

A2L1 = np.float32(LL)
H2L1 = np.float32(LH)
V2L1 = np.float32(HL)
D2L1 = np.float32(HH)

#Fusion start
AfL1 = 0.5*(A1L1+A2L1);
D=(np.abs(H1L1)-np.abs(H2L1))>=0
HfL1=np.multiply(D,H1L1)+np.multiply(np.logical_not(D),H2L1)
D=(np.abs(V1L1)-np.abs(V2L1))>=0
VfL1=np.multiply(D,V1L1)+np.multiply(np.logical_not(D),V2L1)
D=(np.abs(D1L1)-np.abs(D2L1))>=0
DfL1=np.multiply(D,D1L1)+np.multiply(np.logical_not(D),D2L1)

#For inverse swt2 (iswt2)

coeffs3 = AfL1, (HfL1,VfL1,DfL1)
imf = np.uint8( pywt.iswt2(coeffs3,'db2') );

#Display images
plt.figure(dpi=200)
plt.subplot(121)
plt.imshow(img1,cmap = 'gray')
plt.subplot(122)
plt.imshow(img2,cmap = 'gray')
plt.savefig('inputs.png',dpi=200)
plt.show()

plt.figure(dpi=200)
plt.imshow(imf,cmap = 'gray')
plt.savefig('output.png')
plt.show()
