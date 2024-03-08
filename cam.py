import cv2
import uuid
from logs import LOGS


# [CAM FILE]
#
# - open a camera window
# - take a snapshot with press space bar key
# - save snapshot to images folder
# - close camera window with press ESC key
#



cv2.namedWindow("preview")
# if 1 -> frontal cam
# if 0 -> integrated cam
vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on press ESC key
        break
    elif key == 32:  # take snapshot on press space bar key
        filename = f"./images/{uuid.uuid4()}.png"
        cv2.imwrite(filename, frame)
        print(f"{LOGS['CAM']} Screenshot taken.")

vc.release()
cv2.destroyAllWindows() # close window