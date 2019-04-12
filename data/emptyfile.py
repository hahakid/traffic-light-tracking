import os
import glob
path='./hui'

imlist=glob.glob(path+"/*.jpg")
for l in imlist:
    l=l.replace("jpg",'txt')
    
    if not os.path.exists(l):
        print(l)
        fobj=open(l,'w')
        fobj.close()

