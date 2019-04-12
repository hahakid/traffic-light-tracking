import cv2
import os


w=1292
h=964
probthred=0.05

path='../hui/' # image & label path
imgname='hui_00002630.jpg' #image name 
refile='comp4_det_test_trafficlight.txt' # result from detector valid

def getcorner(co):
    xmid,ymid,width,height=float(co[1]),float(co[2]),float(co[3]),float(co[4])
    xmin=int((xmid-width/2.0)*w)    
    xmax=int((xmid+width/2.0)*w)
    ymin=int((ymid-height/2.0)*h)    
    ymax=int((ymid+height/2.0)*h)
    return (xmin,ymin),(xmax,ymax)

#decode the valid mode result, name, prob, xmin, ymin, xmax, ymax detector.c print_detector_detections()
def detectorresult(re,name):
    f=open(re)
    ll=f.readlines()
    f.close()
    rcoord=[]
    for l in ll:
        if l.split(' ')[0]==name:
            label=l.replace('\n','').split(' ')
            prob=float(label[1])
            xmin=int(float(label[2]))
            ymin=int(float(label[3]))
            xmax=int(float(label[4]))
            ymax=int(float(label[5]))
            #print(prob,xmin,xmax,ymin,ymax)
            if prob>probthred:           
                rcoord.append([prob,(xmin,ymin),(xmax,ymax)])
            #rcoord.append([prob,(ymin,xmin),(ymax,xmax)])
    return rcoord


def getresult(name):
    imgpath=os.path.join(path,name)
    img=cv2.imread(imgpath)
    labelpath=imgpath.replace('jpg','txt')
    f=open(labelpath)
    ll=f.readlines()
    f.close()
    for l in ll:
        coord=l.replace('\n','').split(' ')
        up,down=getcorner(coord)
        img=cv2.rectangle(img,up,down,(0,255,0),1)
    
    dcoord=detectorresult(refile,name.replace('.jpg',''))
    #print(dcoord)
    for d in dcoord:
        img=cv2.rectangle(img,d[1],d[2],(0,0,255),1)
        #print(d[0],d[1],d[2])
        img=cv2.putText(img,str(d[0]),d[1],cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
    cv2.imshow("",img)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()

getresult(imgname)

