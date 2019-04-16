from PIL import Image
import cv2, sys, numpy, os
from utils import base64_to_pil_image,pil_image_to_base64
import base64
import io

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = r'C:\Users\Admin\Desktop\shalini_face\database'
model = cv2.face.LBPHFaceRecognizer_create()
(width, height) = (112, 92)
face_cascade = cv2.CascadeClassifier(haar_file)
(images, lables, names, id) = ([], [], {}, 0)

class Makeup_artist(object):
    def __init__(self):
        pass

    def training_dataset(self):
        print('Training...')
        # Create a list of images and a list of corresponding names

        for (subdirs, dirs, files) in os.walk(datasets):
            for subdir in dirs:
                global id
                names[id] = subdir
                subjectpath = os.path.join(datasets, subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + '/' + filename
                    lable = id
                    global images
                    global lables
                    images.append(cv2.imread(path, 0))
                    lables.append(int(lable))
                id += 1

        # (width, height) = (130, 100)
        # Create a Numpy array from the two lists above
        (images, lables) = [numpy.array(lis) for lis in [images, lables]]

        # OpenCV trains a model from the images
        # NOTE FOR OpenCV2: remove '.face'
        model.train(images, lables)

    def apply_makeup(self, img):

        open_cv_image = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
        # open_cv_image = numpy.array(img)
        # open_cv_image = open_cv_image[:, :, ::-1].copy()

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(open_cv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            # Try to recognize the face
            prediction = model.predict(face_resize)
            cv2.rectangle(open_cv_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            print('prediction is: ', prediction)
            if prediction[1] < 120:
                cv2.putText(open_cv_image, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            else:
                cv2.putText(open_cv_image, 'Unknown', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        img = Image.fromarray(open_cv_image, 'RGB')
        return img


