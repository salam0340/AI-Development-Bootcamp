# Import necessary libraries
import cv2
import numpy as np
import math

# Function to detect hand gestures
def detect_hand_gesture():
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video feed
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally (mirror image)
        frame = cv2.flip(frame, 1)

        # Define a region of interest (ROI) where the hand is expected to appear
        roi = frame[100:300, 100:300]

        # Draw a rectangle around the ROI
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

        # Convert the ROI to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (35, 35), 0)

        # Threshold the image to get a binary image (hand in white, background in black)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour (assuming it's the hand)
            contour = max(contours, key=cv2.contourArea)

            # Create a convex hull around the contour
            hull = cv2.convexHull(contour)

            # Draw the contour and convex hull on the ROI
            cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)
            cv2.drawContours(roi, [hull], -1, (0, 0, 255), 2)

            # Compute the convexity defects
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull_indices)

            # Count the number of fingers extended (based on convexity defects)
            if defects is not None:
                finger_count = 0

                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])

                    # Calculate the angles between fingers (to determine if fingers are extended)
                    a = math.dist(start, end)
                    b = math.dist(start, far)
                    c = math.dist(end, far)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                    # If the angle is less than 90 degrees, it's considered a finger
                    if angle <= 90:
                        finger_count += 1
                        cv2.circle(roi, far, 5, (0, 0, 255), -1)

                # Display the number of fingers extended
                cv2.putText(frame, f"Fingers: {finger_count + 1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the video feed with the hand gesture detection
        cv2.imshow("Hand Gesture Recognition", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start hand gesture detection
detect_hand_gesture()
