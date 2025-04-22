import sys
import os
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Constant import RED, YELLOW, IDLE
from Simulation.Board import ConnectFourBoard

class Deep:
    def __init__(self, color=RED, timeout=None, path = "DL/Files/mymodel21.keras"):
        self.name = 'DEEP AI'
        self.color = color
        self.path = path
    
    def set_color(self, color: int):
        """Set the color of the AI."""
        self.color = color

    def solution(game_state):
        board = game_state.board
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == 1:
                    board[i][j] = -1
                if board[i][j] == -1:
                    board[i][j] = 1
        return board

    def get_move(self, game: ConnectFourBoard):
        model_path = os.path.abspath(self.path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = tf.keras.models.load_model(model_path)
        
        board = game.board.copy()
        if self.color == -1:
            for i in range(len(board)):
                for j in range(len(board[i])):
                    if board[i][j] in [-1, 1]:
                        board[i][j] *= -1   
            print("chan", board)

        board_input = np.array(game.board).reshape(-1, 6,7, 1)
        col_probs = model.predict(board_input)[0]
        print("output model", col_probs)
        col = np.argmax(col_probs)

        return col, None
