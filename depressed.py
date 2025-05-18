import cv2
import mediapipe as mp
import numpy as np
import random
import time

class TicTacToe:
    def __init__(self):
        self.board = np.array([[' ' for _ in range(3)] for _ in range(3)])
        self.current_player = 'X'  # Player 1 (human)

    def make_move(self, row, col):
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            return True
        return False

    def computer_move(self):
        empty_cells = [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.board[row][col] = 'O'  # Player 2 (computer)
            return True
        return False

    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        for i in range(3):
            if all(self.board[i, :] == self.current_player) or all(self.board[:, i] == self.current_player):
                return self.current_player
        if all(np.diag(self.board) == self.current_player) or all(np.diag(np.fliplr(self.board)) == self.current_player):
            return self.current_player
        return None

    def is_draw(self):
        return ' ' not in self.board.flatten()

    def reset(self):
        self.board = np.array([[' ' for _ in range(3)] for _ in range(3)])
        self.current_player = 'X'

class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    def detect_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        gesture = None
        position = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                index_tip = hand_landmarks.landmark[8]  # Index finger tip
                index_mcp = hand_landmarks.landmark[5]  # Index finger base
                wrist = hand_landmarks.landmark[0]       # Wrist
                thumb_tip = hand_landmarks.landmark[4]   # Thumb tip
                pinky_tip = hand_landmarks.landmark[20]  # Pinky tip

                # Pointing: Index tip significantly above base and wrist
                if index_tip.y < index_mcp.y and index_tip.y < wrist.y - 0.05:
                    gesture = 'point'
                    position = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
                
                # Open hand: Wide spread between thumb and pinky
                if abs(thumb_tip.x - pinky_tip.x) > 0.15 and abs(thumb_tip.y - pinky_tip.y) < 0.1:
                    gesture = 'open'

        return gesture, position, frame

def draw_board(frame, board, grid_size):
    cell_size = grid_size // 3
    for i in range(1, 3):
        cv2.line(frame, (0, i * cell_size), (grid_size, i * cell_size), (255, 255, 255), 2)
        cv2.line(frame, (i * cell_size, 0), (i * cell_size, grid_size), (255, 255, 255), 2)
    
    for i in range(3):
        for j in range(3):
            if board[i][j] != ' ':
                cv2.putText(frame, board[i][j], (j * cell_size + cell_size // 3, i * cell_size + 2 * cell_size // 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    return frame

def map_position_to_grid(pos, grid_size):
    if pos is None:
        return None
    x, y = pos
    if 0 <= x < grid_size and 0 <= y < grid_size:
        row = y // (grid_size // 3)
        col = x // (grid_size // 3)
        return row, col
    return None

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for performance
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    game = TicTacToe()
    detector = HandGestureDetector()
    grid_size = 300

    selected_cell = None
    last_gesture = None
    last_move_time = 0
    move_cooldown = 1.0  # Seconds between moves to prevent spam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gesture, position, frame = detector.detect_gesture(frame)
        frame = draw_board(frame, game.board, grid_size)
        
        # Show gesture status for debugging
        gesture_text = f"Gesture: {gesture if gesture else 'None'}"
        cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Handle Player 1 (human) moves
        if game.current_player == 'X':
            cell = map_position_to_grid(position, grid_size)
            
            if gesture == 'point' and cell:
                row, col = cell
                cell_size = grid_size // 3
                top_left = (col * cell_size, row * cell_size)
                bottom_right = ((col + 1) * cell_size, (row + 1) * cell_size)
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
                selected_cell = cell
            
            current_time = time.time()
            if (gesture == 'open' and selected_cell and last_gesture != 'open' and 
                current_time - last_move_time > move_cooldown):
                row, col = selected_cell
                if game.make_move(row, col):
                    last_move_time = current_time
                    winner = game.check_winner()
                    if winner:
                        cv2.putText(frame, f"Player {winner} Wins!", (10, grid_size + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Tic-Tac-Toe', frame)
                        cv2.waitKey(2000)
                        game.reset()
                    elif game.is_draw():
                        cv2.putText(frame, "Draw!", (10, grid_size + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Tic-Tac-Toe', frame)
                        cv2.waitKey(2000)
                        game.reset()
                    else:
                        game.switch_player()
        
        # Handle Player 2 (computer) moves
        elif game.current_player == 'O':
            if time.time() - last_move_time > 0.5:  # Small delay for natural feel
                if game.computer_move():
                    last_move_time = time.time()
                    winner = game.check_winner()
                    if winner:
                        cv2.putText(frame, f"Player {winner} Wins!", (10, grid_size + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Tic-Tac-Toe', frame)
                        cv2.waitKey(2000)
                        game.reset()
                    elif game.is_draw():
                        cv2.putText(frame, "Draw!", (10, grid_size + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Tic-Tac-Toe', frame)
                        cv2.waitKey(2000)
                        game.reset()
                    else:
                        game.switch_player()

        last_gesture = gesture
        cv2.putText(frame, f"Player: {'You' if game.current_player == 'X' else 'Computer'}", 
                    (10, grid_size + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Tic-Tac-Toe', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()