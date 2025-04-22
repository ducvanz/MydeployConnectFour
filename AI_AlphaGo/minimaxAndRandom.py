from copy import deepcopy
import random
import numpy as np
import numpy
import math


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Simulation.Board import ConnectFourBoard
from Constant import RED, YELLOW, IDLE 

DEFAULT_WEIGHT = [0.04, 0.02, 0.04, 0.02, 0.03]

class MinimaxAI2:
    def __init__(self, weight=DEFAULT_WEIGHT, depth=6, notPrunning=False, color=RED, timeout=None):
        self.name = 'MinimaxAI depth=' + str(depth)
        self.color = color
        self.depth = depth

        self.notPrunning = notPrunning
        self.weight = { 'allie': [weight[0], weight[1]], 
                        'enemy': [weight[2], weight[3]],
                       'center': weight[4]}
        self.memo = {}
    
    def set_color(self, color: int):
        """Set the color of the AI."""
        self.color = color

    def order_columns(self, game: ConnectFourBoard, valid_columns):
        center = game.columns // 2
        return sorted(valid_columns, key=lambda c: abs(center - c))

    def evaluate_window(self, window, piece):
        """Evaluate a 4-cell window and return a score."""
        score = 0
        opp_piece = -piece  # Opponent piece

        if window.count(piece) == 3 and window.count(IDLE) == 1:
            score += self.weight['allie'][0]
        elif window.count(piece) == 2 and window.count(IDLE) == 2:
            score += self.weight['allie'][1]

        if (window.count(opp_piece)) == 3 and (window.count(IDLE) == 1) :
            score -= self.weight['enemy'][0]
        elif (window.count(opp_piece) == 2) and (window.count(IDLE) == 2) :
            score -= self.weight['enemy'][1]

        return score

    def evaluate(self, game: ConnectFourBoard):
        """Evaluate the board state and return a score."""
        score = 0
        turn = self.color  # Current player (RED or YELLOW)
        
        # Evaluate center column
        center = game.board[:, game.columns // 2]
        score += (np.sum(center == self.color) - np.sum(center == -self.color)) * self.weight['center']  # More pieces in the center is better
        
        # Evaluate horizontal windows
        for r in range(game.rows):
            row_array = [int(i) for i in list(game.board[r, :])]
            for c in range(game.columns - 3):
                window = row_array[c:c + 4]
                score += self.evaluate_window(window, turn)
        
        # Evaluate vertical windows
        for c in range(game.columns):
            col_array = [int(i) for i in list(game.board[:, c])]
            for r in range(game.rows - 3):
                window = col_array[r:r + 4]
                score += self.evaluate_window(window, turn)
        
        # Evaluate diagonal windows (top-left to bottom-right)
        for r in range(game.rows - 3):
            for c in range(game.columns - 3):
                window = [game.board[r + i][c + i] for i in range(4)]
                score += self.evaluate_window(window, turn)
        
        # Evaluate diagonal windows (bottom-left to top-right)
        for r in range(3, game.rows):
            for c in range(game.columns - 3):
                window = [game.board[r - i][c + i] for i in range(4)]
                score += self.evaluate_window(window, turn)
        
        return score

    def minimax(self, game: ConnectFourBoard, depth: int, maximizingPlayer: bool, Alpha:float=-math.inf, Beta:float=math.inf):
        """Minimax algorithm with alpha-beta pruning to find the best move."""

        key = tuple(game.board.flatten())
        if key in self.memo:
            return self.memo[key]

        if game.check_win(-game.turn)  :
            if self.color == -game.turn :
                return [1]
            else :
                return [-1]
        if game.is_full() :
            return [0.0]
        if depth == 0: 
            return [self.evaluate(game)]
        
        valid_columns = game.get_available_columns()

        if maximizingPlayer:  # RED player
            scores = [-1.0] * game.columns      # default value for invavlid_column or bad_evaluate_col. Just mean -1 because we won't need this kind of columns
            alpha = -math.inf
            
            for col in self.order_columns(game, valid_columns):
                # Backup current state
                temp_game = game.copy()

                # Drop piece and recurse
                if temp_game.drop_piece(col):  # Only drop if the column is not full
                    e = self.minimax(temp_game, depth - 1, False, Alpha=alpha)
                    scores[col] = float(np.min(e))

                    # if scores[col] == 1:  # nếu đã thắng chắc, không cần thử các nước khác
                    #     break
                    # if (depth == self.depth) :
                    #     print('Turn 1: ', col, scores[col] / 0.95)
                    
                    # Pruning
                    alpha = max(alpha, scores[col])
                    if (not self.notPrunning) and (alpha > Beta):
                        if scores[col] < 0 :
                            scores[col] *= 0.98
                        else :
                            scores[col] *= 1.02
                        break 
            self.memo[key] = scores
            return np.array(scores)

        else:  # YELLOW player
            scores = [1.0] * game.columns
            beta = math.inf

            for col in self.order_columns(game, valid_columns):
                # Backup current state
                temp_game = game.copy()

                # Drop piece and recurse
                if temp_game.drop_piece(col):
                    e = self.minimax(temp_game, depth - 1, True, Beta=beta)
                    scores[col] = float(np.max(e))
                    # if scores[col] == -1:  # nếu đã thắng chắc, không cần thử các nước khác
                    #     break
                    # if (depth == self.depth - 1) :
                    #     print('Turn 2: ', col, '\n', e, np.argmax(e)) 

                    # Pruning
                    beta = min(beta, scores[col])
                    if (not self.notPrunning) and (Alpha > beta) :
                        if scores[col] < 0 :
                            scores[col] *= 1.02
                        else :
                            scores[col] *= 0.98
                        break
            self.memo[key] = scores
            return np.array(scores)

    def get_move(self, game: ConnectFourBoard):
        """Get the best move for the AI using Minimax."""
        import time
        start_time = time.time()

        filled = np.count_nonzero(game.board)
        adaptive_depth = self.depth
        if filled > 30:
            adaptive_depth = min(self.depth + 2, 8)
        elif filled < 10:
            adaptive_depth = max(self.depth - 2, 4)
        evaluated = self.minimax(game, self.depth, True)
        move = int(np.argmax(evaluated))

        # In thời gian tính toán
        compute_time = time.time() - start_time
        print(f"{self.name} computed move in {compute_time:.2f} seconds")

        return move, evaluated
