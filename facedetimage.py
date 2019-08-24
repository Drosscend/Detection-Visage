import cv2

imagePath = "files/groupe1.jpg" # Chemin de l'image

#https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml
cascadeClassifierPath = "haarcascade_frontalface_alt.xml" # Chemin du Classifier pour le visage

cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)

image = cv2.imread(imagePath)

grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # Conversion de la vidéo en noir et blanc

detectedFaces = cascadeClassifier.detectMultiScale(grayImage) # Détection des visages

for(x,y,width,height) in detectedFaces:
	cv2.rectangle(image, (x, y), (x+width, y+height), (0,255,0), 5)

cv2.imwrite('resultat.jpg', image) # crée un fichier "resultat" avec l'image de imagePath.