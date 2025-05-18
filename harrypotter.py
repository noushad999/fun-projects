import asyncio
import platform
import pygame
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize Pygame and MediaPipe
pygame.init()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Harry Potter: Wizard Duel")
FPS = 60
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)  # Gryffindor red
GREEN = (0, 100, 0)  # Slytherin green
YELLOW = (255, 255, 0)  # Spell flash

# Pixelated Harry Potter (player) and Voldemort (enemy) sprites (8x8)
HARRY_SPRITE = [
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 1, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0]
]

VOLDEMORT_SPRITE = [
    [0, 0, 2, 2, 2, 2, 0, 0],
    [0, 2, 2, 2, 2, 2, 2, 0],
    [2, 2, 0, 0, 0, 0, 2, 2],
    [2, 2, 0, 2, 2, 0, 2, 2],
    [2, 2, 0, 2, 2, 0, 2, 2],
    [2, 2, 0, 0, 0, 0, 2, 2],
    [0, 2, 2, 2, 2, 2, 2, 0],
    [0, 0, 2, 2, 2, 2, 0, 0]
]

# Sound effects (synthetic for Pyodide compatibility)
def create_spell_sound():
    sample_rate = 44100
    duration = 0.2
    freq = 440
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    sound = 0.5 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 5)
    sound = (sound * 32767).astype(np.int16)
    sound = np.repeat(sound[:, np.newaxis], 2, axis=1)  # Stereo
    return pygame.sndarray.make_sound(sound)

def create_hit_sound():
    sample_rate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    sound = 0.5 * np.random.randn(len(t))  # White noise
    sound = (sound * 32767).astype(np.int16)
    sound = np.repeat(sound[:, np.newaxis], 2, axis=1)  # Stereo
    return pygame.sndarray.make_sound(sound)

spell_sound = create_spell_sound()
hit_sound = create_hit_sound()

def draw_sprite(surface, sprite, pos, scale, color_map):
    for y, row in enumerate(sprite):
        for x, pixel in enumerate(row):
            if pixel:
                pygame.draw.rect(surface, color_map[pixel], 
                               (pos[0] + x * scale, pos[1] + y * scale, scale, scale))

def process_pen(image, pen_history):
    # Convert to HSV and detect pen tip (default: red)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    LOWER_COLOR = np.array([0, 120, 70])  # Red lower bound
    UPPER_COLOR = np.array([10, 255, 255])  # Red upper bundle
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # Find contours and track largest
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pen_centroid = None
    pen_motion = 0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:  # Minimum area to filter noise
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                pen_centroid = (cx, cy)
                pen_history.append(cy)  # Fixed: Append cy, not cbounds
                if len(pen_history) == pen_history.maxlen:
                    pen_motion = (pen_history[0] - cy) * 10  # Scale motion
    
    return pen_centroid, pen_motion

def process_frame(image, nod_history, pen_history):
    height, width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    head_tilt = 0
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        head_tilt = (left_eye.y - right_eye.y) * 100
    
    pen_centroid, pen_motion = process_pen(image, pen_history)
    
    return head_tilt, pen_motion, pen_centroid

async def game_loop():
    # Game variables
    player_pos = [200, 400]
    enemy_pos = [600, 400]
    player_health = 100
    enemy_health = 100
    player_spell = False
    enemy_spell = False
    punch_cooldown = 0
    enemy_action_timer = 0
    combo_count = 0
    combo_timer = 0
    screen_shake = 0
    hit_flash = 0
    SPRITE_SCALE = 10
    player_hitbox = pygame.Rect(player_pos[0], player_pos[1], 8 * SPRITE_SCALE, 8 * SPRITE_SCALE)
    enemy_hitbox = pygame.Rect(enemy_pos[0], enemy_pos[1], 8 * SPRITE_SCALE, 8 * SPRITE_SCALE)
    
    # Webcam setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Using keyboard controls (A/D to move, Space to cast spell).")
        use_webcam = False
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        use_webcam = True
    
    # Tracking histories
    nod_history = deque(maxlen=10)  # Unused but kept for compatibility
    pen_history = deque(maxlen=10)
    
    async def update_loop():
        nonlocal punch_cooldown, enemy_action_timer, player_pos, enemy_pos
        nonlocal player_health, enemy_health, player_spell, enemy_spell
        nonlocal player_hitbox, enemy_hitbox, combo_count, combo_timer
        nonlocal screen_shake, hit_flash, use_webcam
        
        # Input handling
        head_tilt = 0
        pen_motion = 0
        pen_centroid = None
        frame = None
        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame. Switching to keyboard controls.")
                use_webcam = False
            else:
                head_tilt, pen_motion, pen_centroid = process_frame(frame, nod_history, pen_history)
                # In-game webcam feed overlay (top-left, 160x120)
                frame_resized = cv2.resize(frame, (160, 120))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                webcam_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                screen.blit(webcam_surface, (0, 0))
                # Second webcam output with pen detection
                if pen_centroid is not None:
                    cv2.rectangle(frame, (int(pen_centroid[0] - 10), int(pen_centroid[1] - 10)),
                                 (int(pen_centroid[0] + 10), int(pen_centroid[1] + 10)), (0, 255, 0), 2)
                cv2.imshow("Wand Detection", frame)
                cv2.waitKey(1)
        
        # Keyboard fallback (always available for override)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a] or keys[pygame.K_d] or keys[pygame.K_SPACE]:
            head_tilt = (keys[pygame.K_a] * -5) + (keys[pygame.K_d] * 5)
            pen_motion = 15 if keys[pygame.K_SPACE] else pen_motion
        
        # Player movement (dodge)
        player_pos[0] = max(0, min(WIDTH - 8 * SPRITE_SCALE, player_pos[0] - head_tilt * 7))
        player_hitbox.x = player_pos[0]
        
        # Player spell (pen motion or space)
        if pen_motion > 10 and punch_cooldown <= 0:
            player_spell = True
            punch_cooldown = 20
            spell_sound.play()
            combo_count += 1
            combo_timer = 30
            damage = 10 + (combo_count * 2)  # Combo increases damage
            if player_hitbox.colliderect(enemy_hitbox):
                enemy_health -= damage
                hit_sound.play()
                hit_flash = 5
                screen_shake = 5
        
        if punch_cooldown > 0:
            punch_cooldown -= 1
        if player_spell and punch_cooldown < 10:
            player_spell = False
        
        if combo_timer > 0:
            combo_timer -= 1
        else:
            combo_count = 0
        
        # Enemy AI (aggressive)
        enemy_action_timer += 1
        if enemy_action_timer > 40:
            action = np.random.choice(['spell', 'dodge'], p=[0.7, 0.3])
            if action == 'spell':
                enemy_spell = True
                if enemy_hitbox.colliderect(player_hitbox):
                    player_health -= 12
                    hit_sound.play()
                    hit_flash = 5
                    screen_shake = 5
            else:
                enemy_pos[0] += np.random.choice([-75, 75])
                enemy_pos[0] = max(400, min(WIDTH - 8 * SPRITE_SCALE, enemy_pos[0]))
            enemy_action_timer = 0
        
        if enemy_spell and enemy_action_timer > 30:
            enemy_spell = False
        
        enemy_hitbox.x = enemy_pos[0]
        
        # Screen shake and hit flash
        shake_offset = [0, 0]
        if screen_shake > 0:
            shake_offset = [np.random.randint(-5, 5), np.random.randint(-5, 5)]
            screen_shake -= 1
        
        if hit_flash > 0:
            screen.fill(YELLOW)
            hit_flash -= 1
        
        # Draw
        screen.fill(BLACK)
        draw_sprite(screen, HARRY_SPRITE, [player_pos[0] + shake_offset[0], player_pos[1] + shake_offset[1]], 
                   SPRITE_SCALE, {1: RED})
        draw_sprite(screen, VOLDEMORT_SPRITE, [enemy_pos[0] + shake_offset[0], enemy_pos[1] + shake_offset[1]], 
                   SPRITE_SCALE, {2: GREEN})
        
        # Health bars
        pygame.draw.rect(screen, RED, (50, 50, player_health * 2, 20))
        pygame.draw.rect(screen, GREEN, (WIDTH - 250, 50, enemy_health * 2, 20))
        
        # Spell indicators
        if player_spell:
            pygame.draw.circle(screen, YELLOW, (int(player_pos[0] + 8 * SPRITE_SCALE), int(player_pos[1])), 15)
        if enemy_spell:
            pygame.draw.circle(screen, YELLOW, (int(enemy_pos[0]), int(enemy_pos[1])), 15)
        
        # Combo counter
        font = pygame.font.SysFont('arial', 24)
        combo_text = font.render(f'Combo: {combo_count}', True, WHITE)
        screen.blit(combo_text, (50, 80))
        
        pygame.display.flip()
        clock.tick(FPS)
        
        # Check game over
        if player_health <= 0 or enemy_health <= 0:
            return

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            
            await update_loop()
            if player_health <= 0 or enemy_health <= 0:
                break
            
            await asyncio.sleep(1.0 / FPS)
    
    finally:
        if use_webcam:
            cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

async def main():
    await game_loop()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())