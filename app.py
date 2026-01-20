# Ref: https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/

import cv2
import time
import math as m
from matplotlib import image
import mediapipe as mp
import numpy as np

# Try to import pygame for sound
try:
    import pygame
    pygame.mixer.init()
    SOUND_ENABLED = True
except:
    SOUND_ENABLED = False
    print("pygame not available - running without sound")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ============ GAMIFICATION SETTINGS ============
BUDDY_LEVEL_THRESHOLDS = [0, 15, 60]  # seconds for level 1, 2, 3
CELEBRATION_INTERVAL = 50  # celebrate every 5 minutes (300 seconds)

def findDistance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    dist = m.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def findAngle(x1, y1, x2, y2):
    """Calculate angle between two points with respect to y-axis."""
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180/m.pi) * theta
    return degree

def overlay_png(background, overlay, x, y):
    h, w = overlay.shape[:2]

    if y+h > background.shape[0] or x+w > background.shape[1]:
        return background

    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3:] / 255.0

    background[y:y+h, x:x+w] = (
        (1 - mask) * background[y:y+h, x:x+w] +
        mask * overlay_img
    ).astype("uint8")

    return background

def calibrate_posture(cap, pose):
    """
    Simple calibration: ask user to sit with good posture for 5 seconds.
    Returns average neck and torso angles as baseline.
    """
    print("\n=== CALIBRATION ===")
    print("Please sit in your BEST, most comfortable posture")
    print("Hold this position for 5 seconds...")
    
    calibration_frames = 0
    target_frames = 5 * 30  # 5 seconds at ~30fps
    
    neck_angles = []
    torso_angles = []
    
    mp_pose_model = mp.solutions.pose
    
    while calibration_frames < target_frames:
        success, image = cap.read()
        if not success:
            continue
        
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if keypoints.pose_landmarks:
            lm = keypoints.pose_landmarks
            lmPose = mp_pose_model.PoseLandmark
            
            # Get landmarks
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
            
            # Calculate angles
            neck_angle = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_angle = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
            
            neck_angles.append(neck_angle)
            torso_angles.append(torso_angle)
            
            calibration_frames += 1
            
            # Show progress
            progress = int((calibration_frames / target_frames) * 100)
            cv2.putText(image, f"Calibrating... {progress}%", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw feedback
            cv2.circle(image, (l_ear_x, l_ear_y), 7, (255, 255, 255), -1)
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, (255, 255, 255), -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, (255, 255, 255), -1)
        
        cv2.imshow('Posture Buddy - Calibration', image)
        cv2.waitKey(1)
    
    # Calculate average baseline
    avg_neck = sum(neck_angles) / len(neck_angles) if neck_angles else 25
    avg_torso = sum(torso_angles) / len(torso_angles) if torso_angles else 10
    
    # Add some tolerance (threshold = baseline + tolerance)
    neck_threshold = avg_neck + 10  # Allow 10 degrees deviation
    torso_threshold = avg_torso + 8  # Allow 8 degrees deviation
    
    print(f"\n✓ Calibration complete!")
    print(f"Your baseline - Neck: {int(avg_neck)}°, Torso: {int(avg_torso)}°")
    print(f"Thresholds - Neck: {int(neck_threshold)}°, Torso: {int(torso_threshold)}°")
    print("\nStarting posture buddy...\n")
    
    time.sleep(2)  # Pause to show message
    
    return neck_threshold, torso_threshold

def play_sound(sound_type):
    """Play a sound if pygame is available."""
    if not SOUND_ENABLED:
        return
    
    try:
        if sound_type == 'celebrate':
            # Try to load and play celebration sound
            # Using relative path for cross-platform compatibility
            sound_path = "D:/IMP Files/C Drive belongings/Dell/Desktop/python-project/posture_ai/assets/celebrate.wav"
            
            # Check if file exists
            import os
            if not os.path.exists(sound_path):
                print(f"⚠️ Sound file not found: {sound_path}")
                print("To enable celebration sounds, add 'celebrate.wav' to the assets folder")
                return
                
            sound = pygame.mixer.Sound(sound_path)
            sound.play()
            print("🔊 Playing celebration sound!")
    except Exception as e:
        print(f"Could not play sound: {e}")


def main():
    # Initialize

    buddy_images = {
    "happy": cv2.imread("D:/IMP Files/C Drive belongings/Dell/Desktop/python-project/posture_ai/assets/happy.png", cv2.IMREAD_UNCHANGED),
    "neutral": cv2.imread("D:/IMP Files/C Drive belongings/Dell/Desktop/python-project/posture_ai/assets/neutral.png", cv2.IMREAD_UNCHANGED),
    "worried": cv2.imread("D:/IMP Files/C Drive belongings/Dell/Desktop/python-project/posture_ai/assets/worried.png", cv2.IMREAD_UNCHANGED),
    "sad": cv2.imread("D:/IMP Files/C Drive belongings/Dell/Desktop/python-project/posture_ai/assets/sad.png", cv2.IMREAD_UNCHANGED),
    "celebrate": cv2.imread("D:/IMP Files/C Drive belongings/Dell/Desktop/python-project/posture_ai/assets/celebrate.png", cv2.IMREAD_UNCHANGED)
}
    
    mp_pose_model = mp.solutions.pose
    pose = mp_pose_model.Pose()
    cap = cv2.VideoCapture(0)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    # Calibration
    neck_threshold, torso_threshold = calibrate_posture(cap, pose)
    
    # Game state
    good_frames = 0
    bad_frames = 0
    total_good_time = 0
    buddy_level = 1
    last_celebration_time = 0
    current_mood = 'neutral'
    celebration_counter = 0
    
    # Time-based tracking (for accurate timing at low fps)
    last_posture_change_time = time.time()  # When posture status changed
    good_posture_start_time = None  # When current good streak started
    bad_posture_start_time = None   # When current bad streak started
    
    # Animation tracking
    level_up_time = None  # When buddy last leveled up
    streak_milestone_time = None  # When user hit a streak milestone
    
    # Colors
    green = (127, 255, 0)
    red = (50, 50, 255)
    white = (255, 255, 255)
    yellow = (0, 255, 255)
    
    print("=== POSTURE BUDDY STARTED ===")
    print("Your buddy is here to help! Keep good posture to make them happy :)")
    
    start_time = time.time()
    
    while True:
        success, image = cap.read()
        if not success:
            print("Camera error")
            break
        
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if not keypoints.pose_landmarks:
            cv2.imshow('Posture Buddy', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        lm = keypoints.pose_landmarks
        lmPose = mp_pose_model.PoseLandmark
        
        # Get landmarks
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
        
        # Calculate angles
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
        
        # Determine posture quality using calibrated thresholds
        is_good_posture = (neck_inclination < neck_threshold and 
                          torso_inclination < torso_threshold)
        
        # Update frames (kept for reference)
        if is_good_posture:
            good_frames += 1
            bad_frames = 0
            # Track good streak start time
            if good_posture_start_time is None:
                good_posture_start_time = time.time()
            bad_posture_start_time = None
        else:
            bad_frames += 1
            good_frames = 0
            # Track bad streak start time
            if bad_posture_start_time is None:
                bad_posture_start_time = time.time()
            good_posture_start_time = None
        
        # Calculate times using actual elapsed time (more accurate at low fps)
        current_time = time.time()
        if good_posture_start_time is not None:
            good_time = current_time - good_posture_start_time
        else:
            good_time = 0
            
        if bad_posture_start_time is not None:
            bad_time = current_time - bad_posture_start_time
        else:
            bad_time = 0
        
        # Update total good time accumulator
        total_good_time += (1 / fps) if is_good_posture else 0
        
        # Determine buddy mood based on CURRENT sustained posture
        if bad_time > 20:  # 20+ seconds of continuous bad posture
            current_mood = 'sad'
        elif bad_time > 15:  # 15-25 seconds of continuous bad posture
            current_mood = 'worried'
        elif bad_time > 0:  # Any bad posture (not sustained yet)
            current_mood = 'neutral'
        elif good_time > 10:  # 10+ seconds of continuous good posture
            current_mood = 'happy'
        else:  # Good posture but not sustained yet
            current_mood = 'neutral'
        
        # Check for celebration (every 5 minutes of total good posture)
        if int(total_good_time) % CELEBRATION_INTERVAL == 0 and int(total_good_time) > last_celebration_time:
            current_mood = 'celebrate'
            celebration_counter = 60  # Show celebration for 60 frames (~2 seconds)
            last_celebration_time = int(total_good_time)
            play_sound('celebrate')
            print(f"🎉 CELEBRATION! You've maintained good posture for {int(total_good_time/60)} minutes!")
        
        # Handle celebration countdown
        if celebration_counter > 0:
            current_mood = 'celebrate'
            celebration_counter -= 1
        
        # Update buddy level based on total good time
        previous_buddy_level = buddy_level
        if total_good_time >= BUDDY_LEVEL_THRESHOLDS[2]:
            buddy_level = 3
        elif total_good_time >= BUDDY_LEVEL_THRESHOLDS[1]:
            buddy_level = 2
        else:
            buddy_level = 1
        
        # Track level-up animation
        if buddy_level > previous_buddy_level:
            level_up_time = time.time()
            print(f"⭐ LEVEL UP! Buddy reached level {buddy_level}!")
        
        # Create and overlay buddy
        # Create and overlay buddy
        buddy_img = buddy_images[current_mood]

        # Base size for buddy (much smaller!)
        base_size = 150  # pixels
        scale = 0.8 + 0.1 * buddy_level
        scale *= 1.0 + 0.03 * m.sin(time.time() * 3)
        
        # Level-up pulse animation (lasts 1.5 seconds)
        if level_up_time is not None:
            time_since_levelup = time.time() - level_up_time
            if time_since_levelup < 1.5:
                # Pulsing effect: grows then shrinks
                pulse = 1.0 + 0.2 * m.sin(time_since_levelup * 8)  # Oscillates rapidly
                scale *= pulse
            else:
                level_up_time = None  # Animation done
        
        # Good streak pulse animation (subtle, continuous)
        if is_good_posture:
            scale *= 1.0 + 0.02 * m.sin(time.time() * 2)  # Subtle breathing effect

        final_size = int(base_size * scale)

        buddy_img = cv2.resize(
            buddy_img,
            (final_size, final_size),
            interpolation=cv2.INTER_AREA
        )

        # Position buddy in top-right corner with proper spacing
        buddy_x = w - final_size - 20  # 20 pixels from right edge
        buddy_y = 20  # 20 pixels from top

        overlay_png(image, buddy_img, buddy_x, buddy_y)
        
        # Draw halo effect during level-up (subtle glow)
        if level_up_time is not None:
            time_since_levelup = time.time() - level_up_time
            if time_since_levelup < 1.5:
                halo_radius = int(final_size / 2 + 10 + 5 * m.sin(time_since_levelup * 6))
                cv2.circle(image, (buddy_x + final_size // 2, buddy_y + final_size // 2), 
                          halo_radius, (100, 255, 255), 2)  # Yellow halo
        
        # Draw minimal UI elements
        # Buddy level indicator
        level_text = f"Buddy Level: {buddy_level}"
        level_y = 20 + final_size + 30  # Below the buddy image
        cv2.putText(image, level_text, (w - final_size - 20, level_y), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2)
        
        # Progress to next level
        if buddy_level < 3:
            next_level_time = BUDDY_LEVEL_THRESHOLDS[buddy_level]
            time_to_next = max(0, next_level_time - total_good_time)
            progress_pct = int((total_good_time / next_level_time) * 100)
            
            # Progress bar
            bar_width = 200
            bar_height = 20
            bar_x = w - 250
            bar_y = 280
            filled_width = int((progress_pct / 100) * bar_width)
            
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         white, 2)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                         green, -1)
            
            cv2.putText(image, f"{progress_pct}%", (bar_x + bar_width + 10, bar_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
        else:
            cv2.putText(image, "MAX LEVEL!", (w - 250, 290), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, yellow, 2)
        
        # Session stats (minimal)
        session_time = int(time.time() - start_time)
        stats_text = f"Session: {session_time//60}m {session_time%60}s"
        cv2.putText(image, stats_text, (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2)
        
        # Current posture status (small indicator)
        status_color = green if is_good_posture else red
        status_text = "Good Posture" if is_good_posture else "Fix Posture"
        cv2.circle(image, (30, 30), 15, status_color, -1)
        cv2.putText(image, status_text, (55, 38), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw colored border based on posture
        border_thickness = 10
        cv2.rectangle(image, (0, 0), (w-1, h-1), status_color, border_thickness)
        
        cv2.imshow('Posture Buddy', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Final stats
    print("\n=== SESSION COMPLETE ===")
    print(f"Total session time: {session_time//60} minutes {session_time%60} seconds")
    print(f"Good posture time: {int(total_good_time//60)} minutes")
    print(f"Buddy reached level: {buddy_level}")
    print(f"\nThanks for working with your buddy! 👋")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
