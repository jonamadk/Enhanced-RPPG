from main_algorithm import FaceDetector
import cv2

if __name__ == "__main__":
    face_detector = FaceDetector("resources/shape_predictor_68_face_landmarks.dat")
    face_detector.main_algorithm(skip_multi=True, find_largest=True)
    # lock = Lock()
    # lock2 = Lock()
    # p1 = Process(target=openGraph, args=((lock),))
    # p2 = Process(target=main, args=((lock2),))
    # p2.start()
    # p1.start()
    # p2.join()
    # p1.join()
