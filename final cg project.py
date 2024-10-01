#finnal code 
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import tkinter as tk
from threading import Thread

# Giving different arrays to handle colour points of different colour
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific colour
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# The kernel to be used for dilation purpose
kernel = np.ones((5,5),np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

class ProjectInfoApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Project Information")
        self.root.geometry("500x500")
        self.root.configure(bg="blue")
        
        # College Name
        self.college_label = tk.Label(root, text="College Name:", font=("Times New Roman", 16, "bold"), bg="blue")
        self.college_label.pack(pady=(20, 0))
        self.college_name = tk.Label(root, text="Sapthagiri College Of Engineering", font=("Times New Roman", 14), bg="blue")
        self.college_name.pack(pady=(0, 10))
        
        # Project Title
        self.project_label = tk.Label(root, text="Project Title:", font=("Helvetica", 16, "bold"), bg="blue")
        self.project_label.pack(pady=(10, 0))
        self.project_title = tk.Label(root, text="Air Canvas Using OpenCV", font=("Times New Roman", 14), bg="blue")
        self.project_title.pack(pady=(0, 10))
        
        # Team Members
        self.team_label = tk.Label(root, text="Team Members:", font=("Helvetica", 16, "bold"), bg="blue")
        self.team_label.pack(pady=(10, 0))
        self.team_members = tk.Label(root, text="Rajeev Kumar\nRajesh Jana\nShashi Vardhan\nRishu Kumar Sureka", font=("Times New Roman", 14), bg="blue")
        self.team_members.pack(pady=(0, 10))
        
        # Guide Name
        self.guide_label = tk.Label(root, text="Guide Name:", font=("Helvetica", 16, "bold"), bg="blue")
        self.guide_label.pack(pady=(10, 0))
        self.guide_name = tk.Label(root, text="Assist Prof. Anuradha Badage", font=("Times New Roman", 14), bg="blue")
        self.guide_name.pack(pady=(10, 20))

        self.start_button = tk.Button(root, text="Start", font=("Helvetica", 14), command=self.start, bg="white")
        self.start_button.pack(side=tk.LEFT, padx=50, pady=(20, 20))

        self.stop_button = tk.Button(root, text="Stop", font=("Helvetica", 14), command=self.stop, bg="white")
        self.stop_button.pack(side=tk.RIGHT, padx=50, pady=(20, 20))

        self.stop_processing = False

    def start(self):
        self.stop_processing = False
        self.thread = Thread(target=self.process_image)
        self.thread.start()

    def stop(self):
        self.stop_processing = True
        self.thread.join()

    def process_image(self):
        global bpoints, gpoints, rpoints, ypoints, blue_index, green_index, red_index, yellow_index, colorIndex
        
        # Here is code for Canvas setup
        paintWindow = np.zeros((471, 636, 3)) + 255
        paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
        paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
        paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
        paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
        paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)

        cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

        # Initialize mediapipe
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        mpDraw = mp.solutions.drawing_utils

        # Initialize the webcam
        cap = cv2.VideoCapture(0)
        ret = True
        while ret and not self.stop_processing:
            # Read each frame from the webcam
            ret, frame = cap.read()

            if not ret:
                break

            x, y, c = frame.shape

            # Flip the frame vertically
            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
            frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
            frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
            frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
            frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
            cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            # Get hand landmark prediction
            result = hands.process(framergb)

            # Post process the result
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        lmx = int(lm.x * 640)
                        lmy = int(lm.y * 480)
                        landmarks.append([lmx, lmy])

                    # Drawing landmarks on frames
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                fore_finger = (landmarks[8][0], landmarks[8][1])
                center = fore_finger
                thumb = (landmarks[4][0], landmarks[4][1])
                cv2.circle(frame, center, 3, (0, 255, 0), -1)
                if (thumb[1] - center[1] < 30):
                    bpoints.append(deque(maxlen=512))
                    blue_index += 1
                    gpoints.append(deque(maxlen=512))
                    green_index += 1
                    rpoints.append(deque(maxlen=512))
                    red_index += 1
                    ypoints.append(deque(maxlen=512))
                    yellow_index += 1
                elif center[1] <= 65:
                    if 40 <= center[0] <= 140: # Clear Button
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]
                        blue_index = 0
                        green_index = 0
                        red_index = 0
                        yellow_index = 0
                        paintWindow[67:,:,:] = 255
                    elif 160 <= center[0] <= 255:
                        colorIndex = 0 # Blue
                    elif 275 <= center[0] <= 370:
                        colorIndex = 1 # Green
                    elif 390 <= center[0] <= 485:
                        colorIndex = 2 # Red
                    elif 505 <= center[0] <= 600:
                        colorIndex = 3 # Yellow
                else:
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(center)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(center)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(center)
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft(center)
            else:
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1

            # Draw lines of all the colors on the canvas and frame
            points = [bpoints, gpoints, rpoints, ypoints]
            for i in range(len(points)):
                for j in range(len(points[i])):
                    for k in range(1, len(points[i][j])):
                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                            continue
                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

            cv2.imshow("Output", frame)
            cv2.imshow("Paint", paintWindow)

            if cv2.waitKey(1) == ord('q'):
                break

        # Release the webcam and destroy all active windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = ProjectInfoApp(root)
    root.mainloop()