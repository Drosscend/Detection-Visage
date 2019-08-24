import cv2
import datetime

# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml
cascadeClassifierPath = 'haarcascade_frontalface_alt.xml' # Chemin du Classifier pour le visage
cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)

cap = cv2.VideoCapture(0) # On récupère la webcam
#cap = cv2.VideoCapture("files/video.mp4") # On récupère une vidéo

while(cap.isOpened()):
	_, frame = cap.read()
	grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Conversion de la vidéo en noir et blanc
	detectedFaces = cascadeClassifier.detectMultiScale(grayImage,  scaleFactor=1.1, minNeighbors=10, minSize=(20, 20)) # Détection des visages

	for(x,y, width, height) in detectedFaces:
		cv2.rectangle(frame, (x, y), (x+width, y+height), (0,255,0), 3) # Dessin d'un rectangle autour du visage
	
	font = cv2.FONT_HERSHEY_SIMPLEX # police d'écriture pour la date
	text = str(datetime.datetime.now()) # texte de la date
	frame = cv2.putText(frame, text, (10,50), font, .7, (0,255,255), 2, cv2.LINE_AA) # Ajout de la date

	cv2.imshow("result", frame)
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()