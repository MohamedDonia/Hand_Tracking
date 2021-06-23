import cv2
import time
import mediapipe as mp

# capture video
cap = cv2.VideoCapture('hand.mp4')

mpHands = mp.solutions.hands
hands = mpHands.Hands()

pTime = time.time()

while True:
    success, img = cap.read()
    # terminate if the image is empty
    if not success:
        break

    # resize image and change color from BGR to RGB
    resized_image = cv2.resize(img, (360, 640))
    imgRGB = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # detect the hands and land marks and draw hand connections
    result = hands.process(imgRGB)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # show some land mark information
            for Id, lm in enumerate(handLms.landmark):
                h, w, c = resized_image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)

                # index finger tip:
                if Id == 8:
                    print(cx, cy)
                    cv2.circle(resized_image, (cx, cy),
                               15, (255, 0, 255),
                               cv2.FILLED)
            # draw land marks and connections
            mp.solutions.drawing_utils.draw_landmarks(resized_image,
                                                      handLms,
                                                      mpHands.HAND_CONNECTIONS)

    # frame rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(resized_image,
                str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 255), 2)
    # show the image
    cv2.imshow("Image", resized_image)
    cv2.waitKey(1)
