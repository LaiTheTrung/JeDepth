# read video file and display
import cv2
import sys
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_video.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    cap = cv2.VideoCapture(video_file)
    count = 0

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break

        cv2.imshow('Video Frame', frame)
        count += 1
        # Press 'q' to exit the video display
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {count}")
# Example usage:
# python3 test_video.py video_record.mp4