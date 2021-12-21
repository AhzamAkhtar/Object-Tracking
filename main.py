import cv2 as cv
capture=cv.VideoCapture("highway.mp4")
# Object Tracking from Stable Camera
object_detector=cv.createBackgroundSubtractorMOG2(history=100,varThreshold=40)
while True:
    ret,frames=capture.read()


    height,width,_=frames.shape
    #print(height,width)
    # Extracting Region of Interest
    roi=frames[340:720,500:800]

    # 1. Object Detection

    mask1=object_detector.apply(roi)
    # Removing Shadow og Gray Color for Better Results
    _,mask=cv.threshold(mask1,254,255,cv.THRESH_BINARY)
    contours,hiarchy=cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # Calculating Area and Removing Small Movement
        area=cv.contourArea(c)
        if area>200:
            #cv.drawContours(roi,c,-1,(0,255,0),2)
            x,y,w,h=cv.boundingRect(c)
            small_roi=frames[y:y+h+500,x:x+w+800]
            cv.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)

    #cv.imshow("roi",roi)
    cv.imshow("Rec",frames)
    #cv.imshow("Mask",mask)
    #cv.imshow("Mask1",mask)
    if cv.waitKey(30)==ord("q"):
        break
cv.release()
cv.destroyAllWindows()