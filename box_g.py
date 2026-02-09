import os
import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import time

# ---------- MediaPipe ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=100, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

# ---------- PyOpenGL Cube ----------
vertices = [
    [1,1,-1],[1,-1,-1],[-1,-1,-1],[-1,1,-1],
    [1,1,1],[1,-1,1],[-1,-1,1],[-1,1,1]
]
edges = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

def Cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

# ---------- Pygame + OpenGL ----------
os.environ['SDL_VIDEO_WINDOW_POS'] = "900,100"
pygame.init()
pygame.display.set_mode((400,400), DOUBLEBUF|OPENGL)
gluPerspective(45, (400/400), 0.1, 50.0)
glTranslatef(0,0,-5)

x_pos, y_pos, z_pos = 0,0,0
x_rot, y_rot, z_rot = 0,0,0

# ---------- Settings ----------
rot_speed = 700
lerp_factor = 0.3
last_distance = 0
hand_detected = False

# Hand animation
hand_animation_start = False
animation_duration = 0.5
animation_start_time = 0

def distance(p1,p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

# ---------- Main Loop ----------
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            quit()

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    target_x, target_y, target_z, target_rot = x_pos, y_pos, z_pos, x_rot

    if result.multi_hand_landmarks:
        handLms = result.multi_hand_landmarks[0]
        hand_detected = True

        # Hand recognition glow at start
        if not hand_animation_start:
            hand_animation_start = True
            animation_start_time = time.time()

        elapsed = time.time() - animation_start_time
        if elapsed < animation_duration:
            overlay = frame.copy()
            for lm in handLms.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                radius = int(10 + 5*math.sin(10*elapsed))
                cv2.circle(overlay,(x,y),radius,(0,140,255),-1)
            frame = cv2.addWeighted(overlay,0.6,frame,0.4,0)

        # --- Cube Gesture Mapping ---
        wrist = handLms.landmark[0]
        thumb = handLms.landmark[4]
        index = handLms.landmark[8]
        middle = handLms.landmark[12]

        # X-axis rotation → thumb-index pinch distance
        dist = distance(thumb,index)
        if abs(dist - last_distance) > 0.008:
            target_rot = dist*rot_speed
        last_distance = dist

        # Y-axis rotation → wrist X coordinate
        y_rot = (wrist.x - 0.5) * 360

        # Z-axis rotation → middle finger X coordinate
        z_rot = (middle.x - 0.5) * 360

        # Cube floating → wrist Y coordinate
        target_x = (wrist.x - 0.5) * 4
        target_y = -(wrist.y - 0.5) * 4
        target_z = -(wrist.y - 0.5) * 2

        # Draw landmarks
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    else:
        hand_detected = False

    if hand_detected:
        # Smooth interpolation
        x_pos += (target_x - x_pos)*lerp_factor
        y_pos += (target_y - y_pos)*lerp_factor
        z_pos += (target_z - z_pos)*lerp_factor
        x_rot += (target_rot - x_rot)*lerp_factor
        # y_rot and z_rot already mapped
    # --- OpenGL Cube Render ---
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glPushMatrix()
    glTranslatef(x_pos, y_pos, z_pos)
    glRotatef(x_rot,1,0,0)
    glRotatef(y_rot,0,1,0)
    glRotatef(z_rot,0,0,1)
    Cube()
    glPopMatrix()

    pygame.display.flip()
    pygame.time.wait(5)

    # --- Webcam feed ---
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
