import cv2
import numpy as np
import time
from squats import pose_module1 as squat_module
from pullups import pose_module as pullup_module

BASE_VIDEO_DIR = './data/train'

choises: dict = dict()
with open('./data/completed_ex.txt', 'r') as file:
    for index, ex in enumerate(file):
        choises[index] = ex
        
# Prompt the user to select an exercise
exercise = input(f"Select the number {[(i, choises[i].strip('\n')) for i in choises]}: ").strip()
exercise = int(exercise)

# Set up video capture (change path if necessary)
cap = cv2.VideoCapture(f"{BASE_VIDEO_DIR}/{choises[exercise]}.mp4")

# Choose the correct pose detector based on the selected exercise
if exercise == 0:
    detector = squat_module.poseDetector()
elif exercise == 1:
    detector = pullup_module.poseDetector()

count = 0
direction = 0  # 0 = going down, 1 = going up
pTime = 0

# Set parameters based on the exercise
if exercise == "squats":
    # For squats: tracking landmarks for hips, knees, and ankles
    landmark1, landmark2, landmark3 = 24, 26, 28  # Right leg (24, 26, 28) or (23, 25, 27)
    angle_range = (68, 171)
else:
    # For pull-ups: tracking landmarks for shoulder, elbow, and wrist
    landmark1, landmark2, landmark3 = 12, 14, 16  # Right arm (12, 14, 16) or (11, 13, 15)
    angle_range = (210, 320)  # Adjusted for typical pull-up motion

while True:
    success, img = cap.read()
    if not success:
        print("Video ended or failed to load.")
        break  # Stop if the video ends

    img = cv2.resize(img, (1280, 720))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        if exercise == 1:
            # Use angle tracking only for pull-ups
            angle = detector.findAngle(img, landmark1, landmark2, landmark3)
            print(f"Detected Angle (Pull-Ups): {angle}")  # Debugging: Print the angle value
            print(f"{count = }")
            
            # Map the angle to a percentage for pull-ups
            per = np.interp(angle, angle_range, (0, 100))
            bar = np.interp(angle, angle_range, (650, 100))

            # Counting logic based on direction and angle
            if per == 0:
                if direction == 0:
                    count += 0.5
                    direction = 1
                    print(f"Incremented count to {count}. Direction set to {direction}.")  # Debugging
            if per == 100:
                if direction == 1:
                    count += 0.5
                    direction = 0
                    print(f"Incremented count to {count}. Direction set to {direction}.")  # Debugging

        else:
            angle = detector.findAngle(img, 24, 26, 28)

            # Переводим углы в проценты, где минимальный угол в присяде равен 0%, а максимальный - 100%
            per = np.interp(angle, (68,171), (0,100))
            bar = np.interp(angle, (68,171), (650, 100))

            if per == 0:
                if direction == 0:
                    count += 0.5
                    direction = 1
            if per == 100:
                if direction == 1:
                    count += 0.5
                    direction = 0

            print(f'{count = }')

        # Draw the progress bar and percentage for pull-ups
        cv2.rectangle(img, (1100, 100), (1175, 650), (34, 255, 0), 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), (34, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

        # Display the current count for both exercises
        cv2.rectangle(img, (0, 450), (250, 720), (34, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        # Optional: Stop the program when count reaches 20
        # if count >= 20:
        #     print("Count limit reached.")
        #     break

    # Show the output image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()