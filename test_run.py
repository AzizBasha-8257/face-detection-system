import cv2

print("OpenCV Version:", cv2.__version__)
print("OpenCV Location:", cv2.__file__)

print("CascadeClassifier:", hasattr(cv2, "CascadeClassifier"))

print("Available Cascade Functions:")

for name in dir(cv2):
    if "Cascade" in name:
        print(name)