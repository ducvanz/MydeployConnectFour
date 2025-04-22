from copy import deepcopy
import random
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Simulation.Board import ConnectFourBoard
from Constant import RED, YELLOW, IDLE 

class Think_One :
    def __init__(self, color=RED, timeout=None):
        self.name = 'Think One'
        self.color = color

    def set_color(self, color:int) :
        self.color = color

    def evaluate(self) :
        # Không sử dụng đánh giá nên mặc định vậy
        return None

    def get_move(self, game:ConnectFourBoard):
        """Simple AI that checks for immediate winning moves.
        
        Args:
            game: The game instance
            
        Returns:
            int: Column index for the move
        """
        if (self.color != game.turn) :
            print('Board và ThinkOne đang bị lệch màu')

        # Try to win in one move
        for i in range(game.columns):
            game_clone = deepcopy(game) 
            if game_clone.drop_piece(i) and game_clone.check_win(self.color):
                return i, self.evaluate()
        
        # If no winning move found, choose a random valid column
        valid_columns = np.where(game.get_available() != -1)[0]
        return random.choice(valid_columns) if len(valid_columns) > 0 else 0, self.evaluate()

