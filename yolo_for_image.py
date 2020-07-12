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


image = cv2.imread("images/football.jpg")
(H, W) = image.shape[:2]


blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
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
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)
            

idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
if len(idxs) > 0:
	for i in idxs.flatten():
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,255), 2)
		text = "{}: {:.2f}".format(labels[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 
              0.5, (0,255,255), 2)

cv2.imshow("Output", image)



















