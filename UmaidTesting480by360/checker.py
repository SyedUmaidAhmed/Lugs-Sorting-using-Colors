import cv2
import numpy as np

lowerBound_green=np.array([145,49,0])
upperBound_green=np.array([180,255,255])

lowerBound_blue=np.array([46,46,0])
upperBound_blue=np.array([161,255,255])

lowerBound_yellow=np.array([0,135,136])
upperBound_yellow=np.array([108,255,255])


kernelOpen = np.ones((5,5))
kernelClose = np.ones((20,20))
kernelOpen1 = np.ones((5,5))
kernelClose1 = np.ones((20,20))
kernelOpen2 = np.ones((5,5))
kernelClose2 = np.ones((20,20))

img = cv2.imread('ryb.jpg')

#convert BGR to HSV

imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# create the Mask

mask=cv2.inRange(imgHSV,lowerBound_green,upperBound_green)
mask1=cv2.inRange(imgHSV,lowerBound_blue,upperBound_blue)
mask2=cv2.inRange(imgHSV,lowerBound_yellow,upperBound_yellow)


#morphology

maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
maskFinal=maskClose

maskOpen1=cv2.morphologyEx(mask1,cv2.MORPH_OPEN,kernelOpen1)
maskClose1=cv2.morphologyEx(maskOpen1,cv2.MORPH_CLOSE,kernelClose1)
maskFinal1=maskClose1

maskOpen2=cv2.morphologyEx(mask2,cv2.MORPH_OPEN,kernelOpen2)
maskClose2=cv2.morphologyEx(maskOpen2,cv2.MORPH_CLOSE,kernelClose2)
maskFinal2=maskClose2

_,conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
_,conts1,h=cv2.findContours(maskFinal1.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
_,conts2,h=cv2.findContours(maskFinal2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


res = cv2.bitwise_and(img, img, mask=mask)


#cv2.drawContours(img,conts,-1,(255,0,0),3)



for contour in conts:
    area1 = cv2.contourArea(contour)
    if(area1 > 300):
        x,y,w,h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(0,0,255),2)
        cv2.putText(img, 'Red Lug', (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 1)
        #cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,255),2)
        #cv2.putText(img, 'Red Lug', (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 1)


for contour in conts1:
    area2 = cv2.contourArea(contour)
    if(area2 > 300):
        x,y,w,h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(255,0,0),2)
        cv2.putText(img, 'Blue Lug', (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 1)        
        #cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
        #cv2.putText(img, 'Blue Lug', (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 1)

for contour in conts2:
    area3 = cv2.contourArea(contour)
    if(area3 > 300):
        x,y,w,h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(0,255,255),2)
        cv2.putText(img, 'Yellow Lug', (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 1)
        #cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        #cv2.putText(img, 'Yellow Lug', (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 1)



cv2.imshow('Lugs', img)
cv2.imshow('COLOR', imgHSV)
#cv2.imshow('Mask', mask2)

cv2.waitKey(0)
cv2.destroyAllWindows()
