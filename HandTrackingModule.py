import cv2
import time
from mediapipe.python.solutions.hands import Hands, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.myhands = Hands(self.mode,
                             self.maxHands,
                             self.detectionCon,
                             self.trackingCon)
        self.results = None

    def findHands(self, img, draw=True):
        # resize image and change color from BGR to RGB
        resized_image = cv2.resize(img, (360, 640))
        imgRGB = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        # detect the hands and land marks and draw hand connections
        self.results = self.myhands.process(imgRGB).multi_hand_landmarks
        if self.results:
            for handLms in self.results:
                # draw land marks and connections
                if draw:
                    draw_landmarks(resized_image,
                                   handLms,
                                   HAND_CONNECTIONS)
        return resized_image

    def findPosition(self, img, handNum=0, draw=True):
        lmList = []
        if self.results:
            myHand = self.results[handNum]
            # show some land mark information
            for Id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([Id, cx, cy])
                # print(cx, cy)
                if draw:
                    cv2.circle(img, (cx, cy),
                               15, (255, 0, 255),
                               cv2.FILLED)

        return lmList


def main():
    pTime = time.time()
    # capture video
    cap = cv2.VideoCapture('hand.mp4')
    detector = handDetector()
    while True:
        success, img = cap.read()
        # terminate if the image is empty
        if not success:
            break
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[4])

        # frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img,
                    str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 0, 255), 2)
        # show the image
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
