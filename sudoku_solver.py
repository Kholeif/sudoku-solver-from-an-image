import cv2
import numpy as np
import imutils
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
img = imutils.resize(img,width=1000)
img2 = img.copy()
img3 = img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.destroyAllWindows()
res=gray

thresh = cv2.adaptiveThreshold(res,255,0,1,19,2)
contour,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
best_cnt = None
for cnt in contour:
    area = cv2.contourArea(cnt)
    if area > 1000:
        if area > max_area:
            max_area = area
            best_cnt = cnt
            
mask = np.zeros((gray.shape),np.uint8)
cv2.drawContours(mask,[best_cnt],0,255,-1)
cv2.drawContours(mask,[best_cnt],0,0,2)
res2 = cv2.bitwise_and(res,mask)

kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

dx = cv2.Sobel(res2,cv2.CV_16S,1,0)
dx = cv2.convertScaleAbs(dx)

close = cv2.morphologyEx(dx,cv2.MORPH_DILATE,kernelx,iterations = 1)

contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if h/w > 10:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)
        
close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
closex = close.copy()

kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))

dy = cv2.Sobel(res2,cv2.CV_16S,0,1)
dy = cv2.convertScaleAbs(dy)

close = cv2.morphologyEx(dy,cv2.MORPH_DILATE,kernely)

contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if w/h > 10:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)

close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
closey = close.copy()

res = cv2.bitwise_and(closex,closey)

contour, hier = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
centroids = []
for cnt in contour:
    mom = cv2.moments(cnt)
    (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
    cv2.circle(img,(x,y),4,(255,0,0),-1)
    centroids.append((x,y))
    
centroids = np.array(centroids,dtype = int)
c = centroids.reshape((100,2))
c2 = c[np.argsort(c[:,1])]

b = np.vstack([c2[i*10:(i+1)*10][np.argsort(c2[i*10:(i+1)*10,0])] for i in range(10)])
bm = b.reshape((10,10,2))

model = cv2.ml.KNearest_load("lallo.model")

def find_k_nearest(model,img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,100,240,cv2.THRESH_BINARY_INV)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    best_cnt=None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

    [x,y,w,h] = cv2.boundingRect(best_cnt)

    if  h>28:
        roi = thresh[y:y+h,x:x+w]
        roismall = cv2.resize(roi,(30,30))
        roismall = roismall.reshape((1,900))
        roismall = np.float32(roismall)
        retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
        string = str(int((results[0][0])))
        return int((results[0][0]))

sudoku = []

for i in range(90):
    if (i+1)%10 == 0:
        continue
    x = (i+1)//10
    y = (i+1)%10 - 1
    my_image = img[bm[x][y][1]:bm[x+1][y+1][1],bm[x][y][0]:bm[x+1][y+1][0]]
    w , h , _ = my_image.shape
    my_image = my_image[w//8:w-w//8,w//8:w-w//8]
    find_k_nearest(model,my_image)
    number = find_k_nearest(model,my_image)
    if number is None:
        sudoku.append(0)
    else:
        sudoku.append(number)
sudoku = np.array(sudoku)
sudoku = sudoku.reshape(9,9)

def check(sudoku,i,j,k):
    for x in range(9):
        if (sudoku[x][j] == k) or (sudoku[i][x] == k):
            return False
        
    x = (i//3)*3
    y = (j//3)*3
    for q in range(x,x+3):
        for w in range(y,y+3):
            if sudoku[q][w] == k:
                return False
    return True

def solve(sudoku):
    global n
    global answer
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] == 0:
                for k in range(1,10):
                    if check(sudoku,i,j,k):
                        sudoku[i][j]=k
                        solve(sudoku)
                        sudoku[i][j]=0
                return
    print("answer",n)
    print(np.matrix(sudoku),"\n")
    answer = np.matrix(sudoku)
    n=n+1
    
n=1
solve(sudoku)

answer = np.array(answer)


for i in range(90):
    if (i+1)%10 == 0:
        continue
    x = (i+1)//10
    y = (i+1)%10 - 1
    if (sudoku[x][y] == 0):
        
        centerX = (bm[x][y+1][0] - bm[x][y][0])/2 + bm[x][y][0]
        centerY = (bm[x+1][y][1] - bm[x][y][1])/2 + bm[x][y][1]
        
        text = str(answer[x][y])
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        
        xx = int(centerX - (textsize[0] / 2))
        yy= int(centerY + (textsize[1] / 2))
        
        cv2.putText(img2, text, (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

cv2.imshow("img",img3)
cv2.imshow("answer",img2)
cv2.waitKey()
cv2.destroyAllWindows()
