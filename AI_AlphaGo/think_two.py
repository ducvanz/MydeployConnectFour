from copy import deepcopy
import random


import sys
import os

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Simulation.Board import ConnectFourBoard
from Constant import RED, YELLOW, IDLE 

class Think_Two:
    def __init__(self, timeout=None, color=RED):
        self.name = 'Think Two'
        self.color = color
        self.timeout = timeout
        
    def set_color(self, color):
        self.color = color
    
    def evaluate(self) :
        return None

    def get_move(self, game:ConnectFourBoard):
        """AI that checks for winning moves and blocks opponent's winning moves.
        
        Args:
            game: The game instance
            
        Returns:
            int: Column index for the move
        """

        if (self.color != game.turn) :
            print('Board và ThinkTwo đang bị lệch màu')
        opponent = -self.color
        valid_columns = np.where(game.get_available() != -1)[0]

        # Try to win in one move
        for i in valid_columns:
            game_clone = deepcopy(game)
            if game_clone.drop_piece(i) and game_clone.check_win(self.color):
                return i, self.evaluate()
        
        # Block opponent's winning move
        for i in valid_columns:
            game_clone = deepcopy(game)
            game_clone.turn = opponent  # Set the turn to opponent for the clone
            if game_clone.drop_piece(i) and game_clone.check_win(opponent):
                return i, self.evaluate()
        
        avoid = []
        # Look ahead to avoid moves that allow opponent to win next turn
        for i in valid_columns:
            game_clone = deepcopy(game)
            if game_clone.drop_piece(i):  # This changes turn to opponent
                # Check if dropping in the same column would give opponent a win
                if game_clone.drop_piece(i) and game_clone.check_win(turn=opponent):
                    avoid.append(i)
        
        if avoid:
            valid_columns = [c for c in valid_columns if c not in avoid]
            return random.choice(valid_columns) if len(valid_columns) > 0 else 0, self.evaluate()

        # If no strategic move found, choose a random valid column
        
        return random.choice(valid_columns) if len(valid_columns) > 0 else 0, self.evaluate()
