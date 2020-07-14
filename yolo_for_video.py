# import the necessary packages
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
vout = None


while(True):
    (grabbed, frame) = vs.read()
    
    if not grabbed:
        break
   
    if W is None or H is None:
        H,W = frame.shape[:2]
        
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    confidences = []
    classIDs = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > 0.50:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                x = centerX - (width/2)
                y = centerY - (height/2)
                
                boxes.append([int(x), int(y), int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.50, 0.30)
    if len(idxs) > 0:
    	for i in idxs.flatten():
    		(x, y) = (boxes[i][0], boxes[i][1])
    		(w, h) = (boxes[i][2], boxes[i][3])
    
    		cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,255), 2)
    		text = "{}: {:.2f}".format(labels[classIDs[i]], confidences[i])
    		cv2.putText(frame, text, (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 
                  0.5, (0,255,255), 2)
    
    if vout is None:
        sz = (frame.shape[1], frame.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        vout = cv2.VideoWriter()
        vout.open('output.avi',fourcc,fps,sz,True)
    
    vout.write(frame)
    # cv2.imshow('Output',frame)
    # if cv2.waitKey(fps) & 0xFF == ord('q'):
    #     break

vs.release()
cv2.destroyAllWindows()