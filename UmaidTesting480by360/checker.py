import cv2
import numpy as np





lowerBound_green=np.array([145,49,0])
upperBound_green=np.array([180,255,255])

lowerBound_blue=np.array([46,46,0])
upperBound_blue=np.array([161,255,255])



kernelOpen = np.ones((5,5))
kernelClose = np.ones((20,20))
kernelOpen1 = np.ones((5,5))
kernelClose1 = np.ones((20,20))


img = cv2.imread('YellowRed.jpg')

#convert BGR to HSV

imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# create the Mask

mask=cv2.inRange(imgHSV,lowerBound_green,upperBound_green)
mask1=cv2.inRange(imgHSV,lowerBound_blue,upperBound_blue)


#morphology

maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
maskFinal=maskClose

maskOpen1=cv2.morphologyEx(mask1,cv2.MORPH_OPEN,kernelOpen1)
maskClose1=cv2.morphologyEx(maskOpen1,cv2.MORPH_CLOSE,kernelClose1)
maskFinal1=maskClose1


_,conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
_,conts1,h=cv2.findContours(maskFinal1.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


res = cv2.bitwise_and(img, img, mask=mask)


#cv2.drawContours(img,conts,-1,(255,0,0),3)



for contour in conts:
    area1 = cv2.contourArea(contour)
    if(area1 > 300):
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(img, 'Red Lug', (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)


for contour in conts1:
    area2 = cv2.contourArea(contour)
    if(area2 > 300):
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        cv2.putText(img, 'Blue Lug', (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)




cv2.imshow('Lugs', img)
cv2.imshow('mask', mask)
cv2.imshow('Mask2', mask1)
cv2.imshow('COLOR', imgHSV)

cv2.waitKey(0)
cv2.destroyAllWindows()
