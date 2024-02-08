import cv2

def main():
    # Open the camera (default camera index is 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()

            # Check if the frame was successfully read
            if not ret:
                print("Error: Could not read frame.")
                break

            # Display the frame on the screen
            cv2.imshow('Camera Feed', frame)

            # Exit the loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release the camera and close the OpenCV window
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
