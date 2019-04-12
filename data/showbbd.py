import numpy as np
import cv2
import glob
w=1292
h=964
threshold=560
img=np.zeros((h,w,3),np.uint8)
img.fill(255)

def getcorner(co):
    xmid,ymid,width,height=float(co[1]),float(co[2]),float(co[3]),float(co[4])
    xmin=int((xmid-width/2.0)*w)    
    xmax=int((xmid+width/2.0)*w)
    ymin=int((ymid-height/2.0)*h)    
    ymax=int((ymid+height/2.0)*h)
    return (xmin,ymin),(xmax,ymax)

plist=glob.glob("./hui/*.jpg")

fourcc=cv2.VideoWriter_fourcc(*'MP4V')
out=cv2.VideoWriter('./drawbb.mp4',fourcc,60,(w,h))

for p in plist:
    txt=p.replace('jpg','txt')
    lfile=open(txt,'r')
    ll=lfile.readlines()
    lfile.close()
    if ll:        
        for l in ll:
            coord=l.replace('\n','').split(' ')
            #print(coord)
            up,down=getcorner(coord)
            print(up,down)                
            img=cv2.rectangle(img,up,down,(1,1,1),1)
            out.write(img)            
            #cv2.imshow("",img)
out.release()            
    #if cv2.waitKey(0) == ord('q'):
    #    cv2.destroyAllWindows()
    #    out.release()        
    #    break
img=cv2.line(img,(0,threshold),(1292,threshold),(0,255,0),3)       
cv2.imwrite('empty.jpg',img)
