# %% [code]
import cv2
import os
import face_recognition

FACES=[]
NAME=[]

KNOWN_FACES_PATH="./images/Webcam"          #path for Known faces in you database           #change path according to your file structure
UNKNOWN_FACES_PATH="./images/unknown"       #path for faces to be recognise                 #change path according to your file structure
MODEL='cnn'                                 #type of model to detect faces
TOLERANCE=0.6                               #how much tolerance is acceptable
FRAME_THICKNESS=1                           #thickness of frame around detected or recognised faces
FONT_THICKNESS=2                            #font thickness in image

print("Loading known faces...")
# Loding our images from database
for name in os.listdir(KNOWN_FACES_PATH):
    for filename in os.listdir(f"{KNOWN_FACES_PATH}/{name}"):
        image=face_recognition.load_image_file(f"{KNOWN_FACES_PATH}/{name}/{filename}")
        encoding=face_recognition.face_encodings(image)[0]
        FACES.append(encoding)
        NAME.append(name)

print("Loading unknows faces...")
# Loading Unknown images
for filename in os.listdir(UNKNOWN_FACES_PATH):
    image=face_recognition.load_image_file(f"{UNKNOWN_FACES_PATH}/{filename}")
    locations=face_recognition.face_locations(image, model=MODEL)
    encodings=face_recognition.face_encodings(image, locations)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    # uncomment below lines for face detection

    for face_location in locations:
        print(f"drawing rectangle")
        top_left=(face_location[3], face_location[0])
        bottom_right=(face_location[2], face_location[1])
        cv2.rectangle(image, top_left, bottom_right, FRAME_THICKNESS)

    # uncomment below lines for face recognition

    # for face_encoding, face_location in zip(encodings, locations):
    #     result=face_recognition.compare_faces(FACES, face_encoding, TOLERANCE)
    #     match=None
    #     if True in result:
    #         match=NAME[result.index(True)]
    #         print(f"Mathch Found")
    #
    #         top_left=(face_location[3], face_location[0])
    #         bottom_right=(face_location[2], face_location[1])
    #         color=[0,255,0]
    #         cv2.rectangle(image,top_left, bottom_right, color, FRAME_THICKNESS)
    #
    #         top_left=(face_location[3], face_location[2])
    #         bottom_right=(face_location[1], face_location[2]+22)
    #         cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
    #         cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)
    cv2.imshow(filename, image)
    cv2.waitKey(10000)
