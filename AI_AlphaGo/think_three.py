from copy import deepcopy
import random
import time
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Simulation.Board import ConnectFourBoard
from Constant import RED, YELLOW, IDLE 

class Think_Three :
    def __init__(self, color=RED, timeout=None) :
        self.color = color
        self.name = 'Think Three'
        self.timeout = timeout

    def set_color(self, color) :
        self.color = color

    def evaluate(self, game, start_time) :
        valid_columns = np.where(self.game.get_available() > -1)[0]

        # Look ahead three moves with timeout
        w3 = []
        first_move_scores:list = [-1] * self.game.columns
        for i in valid_columns :
            first_move_scores[i] = 0
        
        for i in valid_columns:
            # Check if we're running out of time
            if time.time() - start_time > self.timeout * 0.7:
                # If we have some evaluated moves, choose the best
                if w3:
                    # Count occurrences of each first move in winning sequences
                    for move in w3:
                        first_move_scores[move] += 1
                    return max(first_move_scores, key=first_move_scores.get)
                else:
                    # Fall back to random valid move
                    return random.choice(valid_columns)
                    
            try:
                game_clone = deepcopy(self.game)
                if game_clone.drop_piece(i):  # This changes turn to opponent
                    opponent_valid_columns = [c for c in range(self.game.columns) if game_clone.board[0][c] == 0]
                    
                    for j in opponent_valid_columns:
                        # Check if we're running out of time
                        if time.time() - start_time > self.timeout:
                            break
                            
                        game_clone2 = deepcopy(game_clone)
                        if game_clone2.drop_piece(j):  # This changes turn back to original player
                            player_valid_columns = [c for c in range(self.game.columns) if game_clone2.board[0][c] == 0]
                            
                            for k in player_valid_columns:
                                # Final timeout check
                                if time.time() - start_time > self.timeout:
                                    break
                                    
                                game_clone3 = deepcopy(game_clone2)
                                if game_clone3.drop_piece(k) and game_clone3.check_win(self.color):
                                    w3.append(i)
                                    # Once we find a winning path from this first move, we can stop
                                    # exploring other paths from the same first move
                                    break
            except Exception as e:
                print(f"Error in Think Three AI: {e}")
                continue
        
        # Count occurrences of each first move in winning sequences
        for move in w3:
            first_move_scores[move] += 1

        return first_move_scores
    
    def get_move(self, game, max_time=1.0):
        """Enhanced AI that looks ahead three moves.
        
        Args:
            game: The game instance
            max_time: Maximum time in seconds for AI computation
            
        Returns:
            int: Column index for the move
        """
        turn = game.turn  # Use the game's current turn
        opponent = -turn
        
        # Start a timer to prevent excessive computation
        start_time = time.time()
        
        if (self.color != game.turn) :
            print('Board và ThinkThree đang bị lệch màu')
        opponent = -self.color
        valid_columns = np.where(game.get_available() != -1)[0]

        # Try to win in one move
        for i in valid_columns:
            game_clone = deepcopy(game)
            if game_clone.drop_piece(i) and game_clone.check_win(self.color):
                return i, None
        
        # Block opponent's winning move
        for i in valid_columns:
            game_clone = deepcopy(game)
            game_clone.turn = opponent  # Set the turn to opponent for the clone
            if game_clone.drop_piece(i) and game_clone.check_win(opponent):
                return i, None
        
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
            return random.choice(valid_columns) if len(valid_columns) > 0 else 0, None

        # If no strategic move found, choose a random valid column
        return random.choice(valid_columns) if len(valid_columns) > 0 else 0, None
        
        # If only one valid move, return it immediately
        if len(valid_columns) == 1:
            return valid_columns[0]

        # If time is already running out, skip the deep search
        if time.time() - start_time > max_time * 0.3:
            return random.choice(valid_columns), None
        

        
        evalue = self.evaluate(game, start_time=start_time, )

        # If we found winning sequences, choose the first move that leads to the most
        if evalue:
            return evalue.index(max(evalue)), evalue
        
        # If no strategic move found, choose a random valid column
        return random.choice(valid_columns) if valid_columns else 0, None
