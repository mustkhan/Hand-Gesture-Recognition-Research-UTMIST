import datetime
import cv2
import numpy as np

fps_values = []


# Prints average fps
def average_fps():
    total_fps = 0
    for fps in fps_values:
        total_fps += fps
    print(f"Average FPS: {total_fps/len(fps_values)}")


def hand_tracker():
    num_frames = 0
    fps = 0
    start_time = datetime.datetime.now()

    ''' Use Haar Cascade algorithm to detect hand
    and return coordinates of the rectangle around the hand.
    FYI, training took 3.5 hours. '''

    hand_classifier = cv2.CascadeClassifier("classifier/cascade.xml")

    cap = cv2.VideoCapture(0)

    while True:
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time
        fps_values.append(fps)

        average_fps()

        ret, frame = cap.read()
        img = cv2.resize(frame, (600, 300))

        # Convert to binary colour image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Store coordinates of rectangle in list, faces
        hands = hand_classifier.detectMultiScale(
            gray, scaleFactor=1.115, minNeighbors=30)  # Feel free to tinker with this

        try:
            hands = [hands[-1]]
            for x, y, w, h in hands:
                # Light green bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (77, 255, 9), 1)

                # White circle for center point
                cX = (x+x+w)//2
                cY = (y+y+h)//2
                circle = cv2.circle(
                    frame, (cX, cY), 2, (255, 255, 255), -1)
                cv2.putText(frame, f"Center: ({cX}, {cY})", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Light green text indicates fps
                cv2.putText(frame, f"FPS: {int(fps)}", (500, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (77, 255, 9), 2)

        except IndexError:
            pass

        cv2.imshow("Hand Tracker, Haar Cascade", frame)

        # Press Esc to quit
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    hand_tracker()
