import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("")
#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

up = False
down = False
count = 0

with mp_pose.Pose(
    static_image_mode = False) as pose:

    while True:
        ret, frame = cap.read()
        if ret == False:
               break
        #frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks is not None:
            x1 = int(results.pose_landmarks.landmark[15].x * width)
            y1 = int(results.pose_landmarks.landmark[15].y * height)

            x2 = int(results.pose_landmarks.landmark[13].x * width)
            y2 = int(results.pose_landmarks.landmark[13].y * height)

            x3 = int(results.pose_landmarks.landmark[11].x * width)
            y3 = int(results.pose_landmarks.landmark[11].y * height)

            #Angulos
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            p3 = np.array([x3, y3])

            l1 = np.linalg.norm(p2 - p3)
            l2 = np.linalg.norm(p1 - p3)
            l3 = np.linalg.norm(p1 - p2)

            angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))

            #Contador de flex
            if angle >= 165:
                up = True

            if up == True and angle <= 65:
                down = True

            if up == True and down == True and angle >= 165 :
                count += 1
                up = False
                down = False

            #visualization
            aux_image = np.zeros(frame.shape, np.uint8)
            cv2.line(aux_image, (x1, y1), (x2, y2),(255, 255, 0), 20)
            cv2.line(aux_image, (x2, y2), (x3, y3),(255, 255, 0), 20)
            cv2.line(aux_image, (x1, y1), (x3, y3),(255, 255, 0), 5)

            contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
            cv2.fillPoly(aux_image, pts = [contours], color = (255, 100, 0))
            output = cv2.addWeighted(frame, 1, aux_image, 0.6, 0)

            cv2.circle(output, (x1, y1), 6, (0, 255, 255), 4)
            cv2.circle(output, (x2, y2), 6, (0, 255, 255), 4)
            cv2.circle(output, (x3, y3), 6, (0, 255, 255), 4)
            cv2.rectangle(output, (0, 0), (200, 175), (0, 0, 0), -1)  #135, 100 para webcam
            cv2.putText(output, str(int(angle)), (x2 + 30, y2), 2, 2, (255, 100, 0), 2)
            cv2.putText(output, str(count), (5, 155), 4, 7, (250, 100, 0), 2)

            cv2.namedWindow('mostrar', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("mostrar", 540, 720)
            cv2.imshow("mostrar", output)

        #cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()