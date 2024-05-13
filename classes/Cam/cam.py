import cv2

from CONSTANTES import LOGS

class CameraApp:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.vc = cv2.VideoCapture(self.camera_index)

    def capture_snapshot(self):
        if self.vc.isOpened():
            rval, frame = self.vc.read()
            if rval:
                self.take_snapshot(frame)  # Assegurar que frame é passado como argumento
            else:
                print("Failed to capture frame.")
        else:
            print("Failed to open camera.")
        self.close()

    def take_snapshot(self, frame):
        #filename = f"./images/{uuid.uuid4()}.png"
        filename = f"./images/image.png"
        cv2.imwrite(filename, frame)
        print(f"{LOGS['CAM']} Screenshot taken.")

    def close(self):
        self.vc.release()
        print(f"{LOGS['CAM']} Camera has been closed.")

# # Uso da classe CameraApp
# if __name__ == '__main__':
#     cam_app = CameraApp()
#     cam_app.capture_snapshot()
