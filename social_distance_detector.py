# import the necessary packages
from scipy.spatial import distance as dist
import imutils
import numpy as np
import cv2

# define file path
yolo_path = "yolo-coco/"
image_path =  "images/"


# load the COCO class labels
labels_path = yolo_path + "coco.names"
labels = open(labels_path).read().strip().split("\n")


weights_path = yolo_path + "yolov3.weights"
config_path = yolo_path + "yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture("videos/object_detection_test.mp4")
H, W = (None, None)

fps = 30
min_distance = 50
vout = None

def detect_people(frame, net, ln, personIdx=0):
    (H,W) = frame.shape[:2]
    results = []
      
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    centroids = []
    confidences = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if classID == personIdx and confidence > 0.50:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.30, 0.30)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)
            
    return results


while(True):
    (grabbed, frame) = vs.read()
    
    if not grabbed:
        break

    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, personIdx=labels.index("person"))
    violate = set()
    
    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")
        
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                if D[i,j] < min_distance:
                    violate.add(i)
                    violate.add(j)
                    
    for (i, (prob, box, centroid)) in enumerate(results):
        (startX,startY,endX,endY) = box
        (cX,cY) = centroid
        color = (0, 255, 0)
        
        if i in violate:
            color = (0, 0, 255)
            
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)
        
    text = "Social Distance Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,255), 3)


    if vout is None:
        sz = (frame.shape[1], frame.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        vout = cv2.VideoWriter()
        vout.open('output.avi',fourcc,fps,sz,True)
        
    vout.write(frame)

    # cv2.imshow("Frame", frame)
    # key = cv2.waitKey(1) & 0xFF
    
    # if key == ord("q"):
    #     break

vs.release()
cv2.destroyAllWindows()
        
        
        
        
        
        
        