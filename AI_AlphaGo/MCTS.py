
import random
import time
from copy import deepcopy
from math import sqrt, log


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Simulation.Board import ConnectFourBoard
from Constant import RED, YELLOW, IDLE 

class MonteCarloTreeSearch :
    def __init__ (self, color=RED, num_rollouts=1000, temperature=sqrt(2), max_time=1.5) :
        
        self.num_rollouts = num_rollouts
        self.temperature = temperature
        self.time_limit = max_time

        self.color = color
        self.name = 'Monte Carlo Tree Search'

    def set_color(self, color) :
        self.color = color
 
    def get_move(self, game:ConnectFourBoard):
        """Run Monte Carlo Tree Search to find the best move.
        
        Args:
            game: The ConnectFour game instance
            num_rollouts: Maximum number of simulations to run
            temperature: Exploration parameter
            max_time: Maximum time in seconds to run simulations
        
        Returns:
            The best move as column index
        """
        # Get valid columns (non-full columns)
        valid_columns = game.get_available_columns()
        
        # If only one valid move, return it immediately
        # If no valid moves, return a random column
        match len(valid_columns) :
            case 1: 
                return valid_columns[0]
            case 0:
                return random.randint(0, game.columns - 1)

        
        # Initialize statistics
        counts = {move: 0 for move in range(game.columns)}
        wins = {move: 0 for move in range(game.columns)}
        losses = {move: 0 for move in range(game.columns)}
        
        start_time = time.time()
        rollouts_completed = 0
        
        # Run simulations until we reach the limit rollout or timeout
        while rollouts_completed < self.num_rollouts and (time.time() - start_time) < self.time_limit :
            # Select a move to simulate
            move = select(game, self.temperature, counts, wins, losses)
                
            # Expand the game state with the selected move
            game_clone = expand(game, move)
                
            # Skip invalid moves
            if game_clone is None:
                continue
                    
            # Simulate a random game from this state
            reward = simulate(game_clone)
                
            # Update statistics
            counts, wins, losses = backpropagate(game.turn, move, reward, counts, wins, losses)
                
            rollouts_completed += 1

            # print(counts.values(), ' ', wins.values(), ' ', losses.values(), '\n')

        # Choose the best move based on the statistics
        return next_move(game, counts, wins, losses, valid_columns)
    
def expand(game, move:int):
    """Create a new game state by applying the move."""
    game_copy = game.copy()
    # Check if the move is valid before applying
    if game_copy.drop_piece(move):
        return game_copy
    
    # Return None for invalid moves
    return None

def simulate(game_clone:ConnectFourBoard):
    """Simulate a random game from the current state until completion."""
    # Check if the game is already over
    if game_clone.check_win(-game_clone.turn) :
        return -game_clone.turn
        
        # Check for a draw
    if game_clone.is_full() :
        return 0
        
    # Limit simulation to a reasonable number of moves
    move_limit = game_clone.columns * game_clone.rows  # Maximum possible moves in a 6x7 board
    move_count = 0
        
    # Make random moves until the game is over
    while move_count < move_limit:
        # Get all valid columns
        valid_columns = game_clone.get_available_columns()
            
        # If no valid moves, it's a draw
        if len(valid_columns) == 0:
            return 0
                
        move = random.choice(valid_columns)
            
        if game_clone.drop_piece(move):
            # Check if the last move resulted in a win
            if game_clone.check_win(game_clone.turn):
                return game_clone.turn
                    
            # Check for a draw
            if game_clone.is_full() :
                 return 0
                    
            move_count += 1
        else:
            # If move couldn't be made (column full), try another one
            continue
        
    # If we've reached the move limit without a conclusion, return a draw
    return 

def backpropagate(turn, move, reward, counts, wins, losses):
    """Update the statistics for the move based on simulation result."""
    counts[move] = counts.get(move, 0) + 1
    if reward == turn:
        wins[move] = wins.get(move, 0) + 1
    elif reward != 0:  # Only count as loss if not a draw
        losses[move] = losses.get(move, 0) + 1
    return counts, wins, losses

def select(game:ConnectFourBoard, temperature, counts:dict, wins:dict, losses:dict):
    """Select a move based on UCT score."""
    # Calculate the UCT score for all next moves
    scores = {}
    for k in game.get_available_columns():
        # The ones not visited get the priority
        if counts.get(k, 0) == 0 :
            scores[k] = 100000
        else:
            scores[k] = (wins.get(k, 0) - losses.get(k, 0)) / counts[k] + \
            temperature * sqrt(log(sum(counts.values())) / counts[k])
        
    # If no valid moves (all columns full), return a random column
    if not scores:
        return random.randint(0, game.columns - 1)
            
    # Select the next move with the highest UCT score
    return max(scores, key=scores.get)

def next_move(game:ConnectFourBoard, counts, wins, losses, valid_columns):
    """Determine the best move based on statistics."""
    # See which action is most promising
    scores = [0] * game.columns
    for k in valid_columns:
        if k not in counts or counts[k] == 0:
            scores[k] = 0.5
        else:
            # Calculate score as win rate
            scores[k] = 0.5 + (wins.get(k, 0) - losses.get(k, 0)) / counts[k]
        
    if not scores:
        return None, None
    
    # print(scores)
    return scores.index(max(scores)), scores
