# Run Haider Run v1.0.0 - AI Gesture-Controlled Endless Runner
# Made by TeamPAK (Haider, Ismail, Kaif, Talha) | MoreProf Week 2026 | Thomas More University

import pygame, os, random, threading, json
import cv2, numpy as np
from ai_edge_litert.interpreter import Interpreter

pygame.init()
pygame.mixer.init()

# ── PATHS & AI MODEL ─────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# Load the Teachable Machine TFLite model and its class labels
interpreter = Interpreter(model_path=os.path.join(BASE, "model_unquant.tflite"))
interpreter.allocate_tensors()
IN  = interpreter.get_input_details()
OUT = interpreter.get_output_details()
LABELS = [l.strip().split(" ", 1)[-1] for l in open(os.path.join(BASE, "labels.txt"))]

# Shared state — written by webcam thread, read by game loop every frame
gesture_state = {"label": "Neutral", "confidence": 0.0}

def webcam_loop():
    # Continuously capture webcam frames, run AI model, update gesture_state
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: continue
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 127.5 - 1)[np.newaxis]  # normalise to [-1,1]
        interpreter.set_tensor(IN[0]['index'], img)
        interpreter.invoke()
        scores = interpreter.get_tensor(OUT[0]['index'])[0]
        idx = int(np.argmax(scores))
        gesture_state["label"]      = LABELS[idx]
        gesture_state["confidence"] = float(scores[idx])

threading.Thread(target=webcam_loop, daemon=True).start()

# ── COMBINED INPUT ────────────────────────────────────────────────────
# Wraps pygame.key.get_pressed() so gesture "virtual keys" work the same
# way as real keyboard keys — game code never needs to know about AI.
class CombinedInput:
    def __init__(self, real, overrides):
        self._real, self._over = real, overrides
    def __getitem__(self, k):
        return self._over.get(k, False) or self._real[k]

# ── WINDOW & ASSETS ───────────────────────────────────────────────────
SW, SH = 1100, 600
SCREEN = pygame.display.set_mode((SW, SH))
pygame.display.set_caption("Run Haider Run")
icon = pygame.image.load(os.path.join(BASE, "Assets/Other/icon.png")).convert_alpha()
pygame.display.set_icon(icon)

def spr(path, w, h):   # helper: load PNG with alpha and scale
    return pygame.transform.scale(pygame.image.load(path).convert_alpha(), (w, h))

def img(path):         # helper: load image without alpha
    return pygame.image.load(path)

H = "Assets/Character"
RUNNING      = [spr(f"{H}/haider.png",      130, 140)] * 2
JUMPING      =  spr(f"{H}/haider_jump.png", 130, 140)
DUCKING      = [spr(f"{H}/haider_duck.png", 130, 90)]  * 2
DEAD         =  spr(f"{H}/haider_dead.png", 300, 330)
RUN_LARGE    =  spr(f"{H}/haider.png",      220, 240)

SMALL_CACTUS = [img(f"Assets/Cactus/SmallCactus{i}.png") for i in range(1, 4)]
LARGE_CACTUS = [img(f"Assets/Cactus/LargeCactus{i}.png") for i in range(1, 4)]
BIRD         = [img(f"Assets/Bird/Bird{i}.png")           for i in range(1, 3)]
CLOUD        =  img("Assets/Other/Cloud.png")
BG           =  img("Assets/Other/Track.png")

# ── AUDIO ─────────────────────────────────────────────────────────────
A = os.path.join(BASE, "Assets", "Audio")
pygame.mixer.music.load(os.path.join(A, "background.mp3"))
pygame.mixer.music.set_volume(0.5)
SFX_JUMP = pygame.mixer.Sound(os.path.join(A, "jump.wav"));  SFX_JUMP.set_volume(0.7)
SFX_WHY  = pygame.mixer.Sound(os.path.join(A, "running.mp3")); SFX_WHY.set_volume(1.0)
SFX_DEAD = pygame.mixer.Sound(os.path.join(A, "dead.mp3"));  SFX_DEAD.set_volume(1.0)

# ── FONTS ─────────────────────────────────────────────────────────────
F_TITLE   = pygame.font.Font('freesansbold.ttf', 52)
F_SUB     = pygame.font.Font('freesansbold.ttf', 26)
F_SCORE   = pygame.font.Font('freesansbold.ttf', 20)
F_HUD     = pygame.font.SysFont(None, 28)
F_CREDITS = pygame.font.Font('freesansbold.ttf', 18)

# Faint background watermark — rendered once, reused every frame
_WM = pygame.font.Font('freesansbold.ttf', 72).render("RUN HAIDER RUN", True, (0,0,0))
_WM.set_alpha(18)

def draw_wm():
    SCREEN.blit(_WM, ((SW - _WM.get_width()) // 2, 80))

def txt(font, text, color, cx, cy):  # helper: render text centred at (cx, cy)
    s = font.render(text, True, color)
    SCREEN.blit(s, s.get_rect(center=(cx, cy)))

CREDITS_TEXT = "Made by TeamPAK"

# ── LEADERBOARD ───────────────────────────────────────────────────────
SCORES_FILE = os.path.join(BASE, "scores.json")

def load_scores():
    try:
        return json.load(open(SCORES_FILE))
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_score(name, score):
    board = load_scores()
    # If this player already has a higher score on the board, don't change anything
    existing = next((e for e in board if e["name"].lower() == name.lower()), None)
    if existing and existing["score"] >= score:
        return board
    # Otherwise remove their old entry and add the new better score
    board = [e for e in board if e["name"].lower() != name.lower()]
    board.append({"name": name, "score": score})
    board = sorted(board, key=lambda x: x["score"], reverse=True)[:10]
    json.dump(board, open(SCORES_FILE, "w"))
    return board

def draw_leaderboard(cx, cy):
    txt(F_SUB, "Top 10 Leaderboard", (30, 30, 30), cx, cy)
    board = load_scores()
    if not board:
        txt(F_CREDITS, "No scores yet!", (140, 140, 140), cx, cy + 36)
    else:
        for i, entry in enumerate(board):
            txt(F_CREDITS, f"{i+1}.  {entry['name']}  -  {entry['score']}", (60, 60, 60), cx, cy + 38 + i * 26)

    # Reset button — drawn below the entries, returns its rect for click detection
    btn = pygame.Rect(cx - 100, cy + 310, 200, 34)
    pygame.draw.rect(SCREEN, (220, 60, 60), btn, border_radius=6)
    txt(F_CREDITS, "Reset Leaderboard", (255, 255, 255), cx, cy + 327)
    return btn

# ── NAME INPUT SCREEN ─────────────────────────────────────────────────
def name_input():
    # Prompt the player to type their name before the game starts.
    # Returns the entered name string (empty name becomes "Player").
    name = ""
    clock = pygame.time.Clock()
    while True:
        SCREEN.fill((255, 255, 255))
        txt(F_TITLE,   "Enter Your Name",      (30, 30, 30),   SW // 2, 200)
        txt(F_SUB,     "Type and press Enter", (120, 120, 120), SW // 2, 265)

        # Render typed name with a blinking cursor
        display = name + ("|" if (pygame.time.get_ticks() // 500) % 2 == 0 else " ")
        txt(F_TITLE, display, (0, 120, 220), SW // 2, 340)

        txt(F_CREDITS, CREDITS_TEXT, (160, 160, 160), SW // 2, 560)
        pygame.display.update()
        clock.tick(30)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); return "Player"
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RETURN:
                    return name.strip() or "Player"   # fallback if nothing typed
                elif e.key == pygame.K_BACKSPACE:
                    name = name[:-1]
                elif len(name) < 16 and e.unicode.isprintable():
                    name += e.unicode

# ── GAME OBJECTS ──────────────────────────────────────────────────────
class Player:
    X, Y, Y_DUCK, VEL = 80, 276, 326, 11

    def __init__(self):
        self.ducking = self.jumping = False
        self.running = True
        self.step    = 0
        self.vel     = self.VEL
        self.image   = RUNNING[0]
        self.rect    = self.image.get_rect(x=self.X, y=self.Y)

    def update(self, inp):
        # Animate and move based on current state
        if self.ducking:
            self.image = DUCKING[self.step // 5]
            self.rect  = self.image.get_rect(x=self.X, y=self.Y_DUCK)
        elif self.running:
            self.image = RUNNING[self.step // 5]
            self.rect  = self.image.get_rect(x=self.X, y=self.Y)
        elif self.jumping:
            self.image = JUMPING
            self.rect.y -= self.vel * 4
            self.vel -= 0.8
            if self.vel < -self.VEL:   # landed
                self.jumping = False
                self.running = True
                self.vel = self.VEL
        self.step = (self.step + 1) % 10

        # State transitions from input
        if inp[pygame.K_UP] and not self.jumping:
            self.running = self.ducking = False
            self.jumping = True
        elif inp[pygame.K_DOWN] and not self.jumping:
            self.running = self.jumping = False
            self.ducking = True
        elif not (self.jumping or inp[pygame.K_DOWN]):
            self.ducking = self.jumping = False
            self.running = True

    def draw(self):
        SCREEN.blit(self.image, self.rect)


class Cloud:
    def __init__(self):
        self.x = SW + random.randint(800, 1000)
        self.y = random.randint(50, 100)

    def update(self):
        self.x -= game_speed
        if self.x < -CLOUD.get_width():
            self.x = SW + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self):
        SCREEN.blit(CLOUD, (self.x, self.y))


class Obstacle:
    def __init__(self, images, y):
        self.images = images
        self.type   = random.randint(0, len(images) - 1)
        self.rect   = images[self.type].get_rect(x=SW, y=y)
        self.index  = 0   # used by Bird for wing animation

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self):
        # Birds cycle between two frames for flapping; others stay static
        frame = (self.index // 5) % len(self.images) if len(self.images) == 2 else self.type
        SCREEN.blit(self.images[frame], self.rect)
        self.index += 1


# ── SCREENS ───────────────────────────────────────────────────────────
def start_screen():
    # Left column (x=290): title, tagline, character, prompt
    # Right column (x=820): leaderboard
    while True:
        SCREEN.fill((255, 255, 255))

        # Left — game info
        txt(F_TITLE,   "RUN HAIDER RUN",                                  (30,30,30),    290,  70)
        txt(F_SUB,     "Gesture-controlled — no keyboard needed",         (100,100,100), 290, 130)
        SCREEN.blit(RUN_LARGE, (290 - RUN_LARGE.get_width()//2,          165))
        if (pygame.time.get_ticks() // 600) % 2 == 0:
            txt(F_SUB, "Press any key to start", (60,60,60),              290, 470)
        txt(F_CREDITS, CREDITS_TEXT, (160,160,160),                       290, 550)

        # Divider
        pygame.draw.line(SCREEN, (220,220,220), (580, 40), (580, 570), 1)

        # Right — leaderboard
        reset_btn = draw_leaderboard(820, 200)

        pygame.display.update()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:  pygame.quit(); return
            if e.type == pygame.MOUSEBUTTONDOWN and reset_btn.collidepoint(e.pos):
                json.dump([], open(SCORES_FILE, "w"))   # wipe the scores file
            if e.type == pygame.KEYDOWN:
                player_name = name_input()
                main(player_name); return


def game_over(player_name):
    board = save_score(player_name, points)
    on_board = any(e["name"] == player_name and e["score"] == points for e in board)

    # Left column (x=290): GAME OVER, score, dead sprite
    # Right column (x=820): leaderboard
    while True:
        SCREEN.fill((255, 255, 255))

        # Left — result
        txt(F_TITLE, "GAME OVER",        (200,30,30),   290, 60)
        txt(F_SUB,   f"Score: {points}", (30,30,30),    290, 120)
        if on_board:
            txt(F_SCORE, "New leaderboard entry!", (0,150,80), 290, 155)
        SCREEN.blit(DEAD, (290 - DEAD.get_width()//2,   175))
        if (pygame.time.get_ticks() // 600) % 2 == 0:
            txt(F_SUB, "Press any key to play again", (60,60,60), 290, 530)
        txt(F_CREDITS, CREDITS_TEXT, (160,160,160),     290, 565)

        # Divider
        pygame.draw.line(SCREEN, (220,220,220), (580, 40), (580, 570), 1)

        # Right — updated leaderboard
        reset_btn = draw_leaderboard(820, 200)

        pygame.display.update()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:  pygame.quit(); return
            if e.type == pygame.MOUSEBUTTONDOWN and reset_btn.collidepoint(e.pos):
                json.dump([], open(SCORES_FILE, "w"))   # wipe the scores file
            if e.type == pygame.KEYDOWN:
                player_name = name_input()
                main(player_name); return


# ── MAIN GAME LOOP ────────────────────────────────────────────────────
def main(player_name="Player"):
    global game_speed, points, obstacles

    player     = Player()
    cloud      = Cloud()
    game_speed = 14
    points     = 0
    obstacles  = []
    x_bg       = 0
    clock      = pygame.time.Clock()

    # Audio state tracking
    was_jumping    = False
    last_milestone = 0

    pygame.mixer.music.play(-1)  # start background music, loop forever

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); return

        SCREEN.fill((255, 255, 255))
        draw_wm()   # faint watermark behind everything

        # ── AI GESTURE → INPUT ───────────────────────────────────────
        g, c = gesture_state["label"], gesture_state["confidence"]
        inp = CombinedInput(pygame.key.get_pressed(), {
            pygame.K_UP:   g == "Jump" and c > 0.80,
            pygame.K_DOWN: g == "Duck" and c > 0.80,
        })

        # ── PLAYER ───────────────────────────────────────────────────
        player.update(inp)
        player.draw()

        # Jump sound — fires once at the rising edge of each jump
        if player.jumping and not was_jumping:
            SFX_JUMP.play()
        was_jumping = player.jumping

        # ── OBSTACLES ────────────────────────────────────────────────
        if not obstacles:
            r = random.randint(0, 2)
            obstacles.append(Obstacle(
                SMALL_CACTUS if r == 0 else LARGE_CACTUS if r == 1 else BIRD,
                325          if r == 0 else 300          if r == 1 else 250
            ))

        for obs in obstacles:
            obs.draw()
            obs.update()
            if player.rect.inflate(-35,-35).colliderect(obs.rect.inflate(-30,-30)):
                pygame.mixer.music.stop()
                SFX_DEAD.play()
                pygame.time.delay(2000)
                game_over(player_name); return

        # ── SCROLLING BACKGROUND ─────────────────────────────────────
        bw = BG.get_width()
        SCREEN.blit(BG, (x_bg, 380))
        SCREEN.blit(BG, (x_bg + bw, 380))
        x_bg = 0 if x_bg <= -bw else x_bg - game_speed

        cloud.update(); cloud.draw()

        # ── SCORE ────────────────────────────────────────────────────
        points += 1
        if points % 200 == 0: game_speed += 1
        txt(F_SCORE, f"Points: {points}", (0,0,0), 1000, 40)

        # "Why are you running" — plays once at every 500-point milestone
        m = points // 500
        if m > last_milestone:
            SFX_WHY.play(); last_milestone = m

        # ── GESTURE HUD ──────────────────────────────────────────────
        color = (0,180,0) if g=="Jump" else (0,120,220) if g=="Duck" else (50,50,50)
        SCREEN.blit(F_HUD.render(f"Gesture: {g}  ({c:.0%})", True, color), (10, 10))
        pygame.draw.rect(SCREEN, (200,200,200), (10, 36, 200, 10))
        pygame.draw.rect(SCREEN, color,         (10, 36, int(200*c), 10))

        clock.tick(30)
        pygame.display.update()


# ── ENTRY POINT ───────────────────────────────────────────────────────
start_screen()