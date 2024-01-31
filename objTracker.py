import argparse
from tools.KalmanFilter import KalmanFilter
from tools.detector import detect
import cv2

def track_objects(video_path):
    sample_time = 0.1
    accel_x, accel_y, noise_std_dev = 1, 1, 1
    meas_noise_x, meas_noise_y = 0.1, 0.1
    tracker = KalmanFilter(sample_time, accel_x, accel_y, noise_std_dev, meas_noise_x, meas_noise_y)
    video_source = cv2.VideoCapture(video_path)

    previous_centers = []
    predicted_states = []

    while video_source.isOpened():
        ret, frame = video_source.read()
        if not ret:
            break

        centers = detect(frame)
        for center in centers:
            tracker.predict()
            prediction = tracker.x
            tracker.update(center)
            
            cv2.circle(frame, (int(center[0]), int(center[1])), 3, (0, 255, 0), -1)
            cv2.rectangle(frame, (int(prediction[0]) - 15, int(prediction[1]) - 15),
                          (int(prediction[0]) + 15, int(prediction[1]) + 15), (255, 0, 0), 2)
            cv2.rectangle(frame, (int(tracker.x[0]) - 15, int(tracker.x[1]) - 15),
                          (int(tracker.x[0]) + 15, int(tracker.x[1]) + 15), (0, 0, 255), 2)

            if previous_centers:
                for i in range(1, len(previous_centers)):
                    cv2.line(frame, (int(previous_centers[i-1][0]), int(previous_centers[i-1][1])),
                             (int(previous_centers[i][0]), int(previous_centers[i][1])), (0, 255, 0), 2)

            if len(predicted_states) > 1:
                cv2.line(frame, (int(predicted_states[-1][0]), int(predicted_states[-1][1])),
                         (int(prediction[0]), int(prediction[1])), (0, 0, 255), 2)

            previous_centers.append(center.flatten())
            predicted_states.append(prediction.flatten())

            cv2.imshow('Object Tracking Frame', frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

    video_source.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Object Tracking using Kalman Filter.')
    parser.add_argument('video_path', type=str, help='The path to the input video file.')
    args = parser.parse_args()

    track_objects(args.video_path)

if __name__ == '__main__':
    main()
