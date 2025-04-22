
import os
import pygame as pg
import time
import argparse
import random
from enum import Enum
import traceback
import numpy as np
import pickle


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))         # Để import file từ folder khác

from Constant import RED, YELLOW, IDLE,  WIDTH, FIRST_MOVING
from Simulation.Board import ConnectFourBoard
from AI_AlphaGo.think_one import Think_One
from AI_AlphaGo.think_two import Think_Two
from AI_AlphaGo.think_three import Think_Three
from AI_AlphaGo.minimaxVsABPrunning import MinimaxAI
from AI_AlphaGo.minimaxAndMCTS import minimaxAndMcts
from AI_AlphaGo.minimaxAndRandom import MinimaxAI2
from AI_AlphaGo.Deep import Deep
from AI_AlphaGo.minimaxDepthInc import minimaxDepthInc

from AI_AlphaGo.MCTS import MonteCarloTreeSearch

from Simulation.Human import Hugeman

class Colors(Enum):
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)

class MatchMaker:

    ###
    #   Thực hiện vẽ giao diện. Và hỗ trợ tương tác người dùng (Human Player)
    #
    #   Attributes:
    #       screen  :   cửa sổ giao diện
    #           width   :   chiều dài cửa số hiển thị được vẽ ra (các kích thước còn lại sẽ tự được chia tương quan phù hợp)
    #           square  :   kích thước 1 ô để vẽ piece          Không khai báo
    #           radius  :   kích thước 1 ô để vẽ piece          Không khai báo
    #           height  :   chiều cao cửa sổ hiển thị           Không khai báo
    #       display_game   :    hiển thị cửa sổ tương tác (True/False)
    #       delay          :    thời gian gap giữa các lượt (turn), để dễ quan sát nếu display
    #       games          :    số lượng trận đấu được tiến hành.
    #
    #       player1 :   người chơi đầu tiên, đánh trước mặc định mang màu FIRST_MOVING
    #       player2 :   người chơi đánh thứ 2
    #
    #   Notices:
    #       Player cần có một số attribute sau:
    #           Player.color        :       màu do người chơi này sử dụng.
    #           Player.name         :       tên người chơi, phục vụ cho hiển thị giao diện.
    #           Player.time_out     :       thời gian tối đa để đưa ra nước đi. MatchMaker bỏ qua việc tính timeout, nên player core tự khai báo và tính toán.
    #       Player cần có một số attribute sau:
    #           Player.set_color(int)             :       setup màu cho player, đề phòng khi cả 2 mặc định chơi cùng 1 màu.
    #           Player.evaluate()                 :       kết quả đánh giá lựa chọn. Nếu ko thực hiện đánh giá thì 'return None'. Mặc định là cột vừa đi bằng 1, còn lại bằng 0.
    #           Player.get_move(ConnectFourBoard) :       return column, evaluate()     ->      Cột muốn đánh và kết quả đánh giá, từ input(Board.ConnectFourBoard)
    ###
    
    def __init__(self,
                 player1, player2,
                 shape=(7,7),  
                 display_game=True, delay=0.5, games=1, 
                 width = WIDTH,
                 train_export_path:str =None,
                 label_export_path:str =None,
                 sleep_between_games=1,
                 display_turn_runtime=True):
        """Initialize the AI vs AI game runner.
        
        Args:
            ai1_level: Difficulty level for AI 1 (Red)
            ai2_level: Difficulty level for AI 2 (Yellow)
            display_game: Whether to show the game graphically
            delay: Delay between moves in seconds (for visualization)
            games: Number of games to play
            ai1_timeout: Maximum time in seconds for AI 1 to compute a move
            ai2_timeout: Maximum time in seconds for AI 2 to compute a move
        """

        self.game = ConnectFourBoard(first_to_move=FIRST_MOVING, save_history=True)

        # Setup user interface window
        self.width = width
        self.square = self.width // self.game.columns
        self.height = (self.game.rows + 1) * self.square
        self.radius = self.square // 2 - 5

        # Setup interaction
        self.display_game = display_game
        self.delay = delay
        self.games = games
        self.display_turn_runtime = display_turn_runtime
        self.sleep_between_games = sleep_between_games
        
        # Setup player
        self.player1 = player1
        self.player2 = player2
        
        self.stats = {"ai1_wins": 0, "ai2_wins": 0, "draws": 0}
        self.history_games = []

        
        # Initialize pygame if needed
        if self.display_game:
            pg.init()
            self.screen = pg.display.set_mode((self.width, self.height))
            pg.display.set_caption(f"AI vs AI - {self.player1.name} vs {self.player2.name}")
    
    def draw_game(self):
        """Draw the current game state on the screen."""
        if not self.display_game:
            return
            
        self.screen.fill(Colors.WHITE.value)
        
        # Draw turn indicator
        font = pg.font.Font(None, 40)
        ai1_name = f"Red ({self.player1.name})"
        ai2_name = f"Yellow ({self.player2.name})"
        text_surface = font.render(f"Turn: {ai1_name if self.game.turn == 1 else ai2_name}", True, Colors.BLACK.value)
        self.screen.blit(text_surface, (20, 20))
        
        # Draw board and pieces
        for c in range(self.game.columns):
            for r in range(self.game.rows):
                pg.draw.rect(self.screen, Colors.BLUE.value, 
                            (c * self.square, (r + 1) * self.square, self.square, self.square))
                color = Colors.BLACK.value
                if self.game.board[r][c] == RED:
                    color = Colors.RED.value
                elif self.game.board[r][c] == YELLOW:
                    color = Colors.YELLOW.value
                pg.draw.circle(self.screen, color, 
                            (c * self.square + self.square // 2, 
                            (r + 1) * self.square + self.square // 2), 
                            self.radius)
        pg.display.update()
    
    def show_win_notification(self, winner):
        """Display the winner notification."""
        if not self.display_game:
            return
            
        font = pg.font.Font(None, 64)
        if winner == 0:
            text = "Draw!"
            text_color = Colors.BLUE.value
        else:
            ai_name = self.player1.name if self.player1.color == winner else self.player2.name
            ai_color = winner
            text = f"{'Red' if ai_color == RED else 'Yellow'} ({ai_name}) Wins!"
            text_color = Colors.RED.value if winner == 1 else Colors.YELLOW.value
            
        text_surface = font.render(text, True, text_color)
        # Add text shadow for better visibility
        shadow_surface = font.render(text, True, Colors.BLACK.value)
        shadow_rect = shadow_surface.get_rect(center=(self.width // 2 + 2, self.square // 2 + 2))
        text_rect = text_surface.get_rect(center=(self.width // 2, self.square // 2))
        self.screen.blit(shadow_surface, shadow_rect)
        self.screen.blit(text_surface, text_rect)
        pg.display.update()
    
    def show_stats(self):
        """Display statistics after all games."""
        ai1_name = self.player1.name
        ai2_name = self.player2.name
            
        self.screen.fill(Colors.WHITE.value)
        font = pg.font.Font(None, 50)
        
        title = f"Results after {self.games} games:"
        text1 = f"Red ({ai1_name}): {self.stats['ai1_wins']} ({self.stats['ai1_wins']/self.games*100:.1f}%)"
        text2 = f"Yellow ({ai2_name}): {self.stats['ai2_wins']} ({self.stats['ai2_wins']/self.games*100:.1f}%)"
        text3 = f"Draws: {self.stats['draws']} ({self.stats['draws']/self.games*100:.1f}%)"
        text4 = "Press any key to exit"
        
        y_pos = self.height // 2 - 100
        for text in [title, text1, text2, text3, text4]:
            text_surface = font.render(text, True, Colors.BLACK.value)
            text_rect = text_surface.get_rect(center=(self.width // 2, y_pos))
            self.screen.blit(text_surface, text_rect)
            y_pos += 50
        
        pg.display.update()
        
        waiting = True
        while waiting and self.display_game:
            for event in pg.event.get():
                if event.type in [pg.QUIT, pg.KEYDOWN]:
                    waiting = False
    
    def play_game(self):
        """Play a single game between two AI agents."""
        self.game = ConnectFourBoard(first_to_move=FIRST_MOVING, save_history=True)
        # Reset game state
        self.player1.set_color(FIRST_MOVING)
        self.player2.set_color(-FIRST_MOVING)
        self.game.reset_game(firstMoving = FIRST_MOVING)
        
        self.draw_game()
        winner = 0
        game_over = False
        
        # Main game loop. Each TURN
        current_turn = FIRST_MOVING
        while not game_over:
            move_start_time = time.time()
            current_player = self.player1 if (current_turn == self.player1.color) else self.player2

            if (self.game.turn != current_player.color) :
                print(f'Bị lệch màu giữa ConnectFourBoard {self.game.turn} và {current_player.name} {current_player.color} trong khi Current_turn={current_turn}')
            
            # Use a timeout for the AI move to prevent hanging
            move_start_time = time.time()
            try:
                # Update caption to show thinking state
                if self.display_game:
                    pg.display.set_caption(f"AI vs AI - {current_player.name} is thinking...")

                col, evaluate = current_player.get_move(self.game)
                valid_columns = self.game.get_available_columns()
                if len(valid_columns) == 0:
                    game_over = True
                    self.stats["draws"] += 1
                    continue
                while col not in valid_columns:
                    col = np.random.choice(valid_columns)

                if col is None:
                    return None         # Quit game

                # Keep responsive
                if self.display_game:
                    for event in pg.event.get():
                        if event.type == pg.QUIT:
                            return None    

                # Validate the move before applying
                if (col < 0) or (col >= self.game.columns) or (self.game.board[0, col] != 0):
                    msg = f'Invalid move response from {current_player.name} at column: {col}'
                    print(self.game.board)
                    raise Exception(msg)
                    
            except Exception as e:
                print(self.game.board)
                print(f"Error in AI move generation: {traceback.TracebackException.from_exception(e)}")
                traceback.print_exc()
                print('Randomize move will kick in')
                time.sleep(0)

                valid_columns = self.game.get_available_columns()
                if len(valid_columns) != 0:
                    col = random.choice(valid_columns)
                    evaluate = np.zeros((self.game.columns,))
                    evaluate[col] = 1
                else:
                    # No valid moves cause board is full. Game is a draw
                    game_over = True
                    self.stats["draws"] += 1
                    continue

            # Reset caption
            if self.display_game and self.games > 1:
                ai1_name = self.player1.name
                ai2_name = self.player2.name
                pg.display.set_caption(f"AI vs AI - {ai1_name} vs {ai2_name}")
                    
            if self.game.drop_piece(col, evaluate):  # This also toggles the turn
                self.draw_game()
                a = 1

            # Check for win
            if self.game.check_win(current_turn):
                winner = current_turn
                game_over = True
                if winner == self.player1.color :
                    self.stats["ai1_wins"] += 1
                else:
                    self.stats["ai2_wins"] += 1

            # Check for draw
            elif self.game.is_full():
                game_over = True
                self.stats["draws"] += 1

            # Record actual compute time for debugging
            # if self.display_turn_runtime :
            #     compute_time = time.time() - move_start_time
            #     if compute_time > 0.0:  # Only report if it took more than the threshold
            #         print(f"{current_player.name[:40].ljust(40)} computed move in {compute_time:.2f}s")

            # Add delay for visualization
            if self.display_game and self.delay > 0:
                time.sleep(self.delay)
                
            current_turn = -current_turn



        # Show the final state and winner
        if self.display_game:
            self.show_win_notification(winner)
            time.sleep(self.sleep_between_games)  # Give some time to see the winner

        return winner
    
    def run(self):
        """Run multiple games between the AI agents."""
        ai1_name = self.player1.name
        ai2_name = self.player2.name
        
        for i in range(1, self.games+1):
            print("ne" , i)
            start_time = time.time()
            if self.display_game :
                pg.display.set_caption(f"AI vs AI - Game {i}/{self.games} - {ai1_name} vs {ai2_name}")
            
            result = self.play_game()
            self.history_games.append(self.game)
            if result is None:  # User quit
                break
                
            playtime = time.time() - start_time
            playturn = np.sum(self.game.board != IDLE)
            # Print progress if not displaying
            if (not self.display_game) :
                print(f"Game {i:>4}/{self.games:>4} complete. " +
                    f"Red ({ai1_name}): {self.stats['ai1_wins']:>3}, " +
                    f"Yellow ({ai2_name}): {self.stats['ai2_wins']:>3}, " +
                    f"Draws: {self.stats['draws']:>3}, " +
                    f"running {playturn} turn in {playtime} seconds")
                
            # if self.train_export_path is not None :
            #     if winner != 0 :
            #         self.game.export_history(winner, train_file_path=self.train_export_path,
            #                                         label_file_path=self.label_export_path)
                

        if self.display_game :
            self.show_stats()


        # print(f"\nResults after {self.games} games:")
        # print(f"Red ({ai1_name}): {self.stats['ai1_wins']} ({self.stats['ai1_wins']/self.games*100:.1f}%)")
        # print(f"Yellow ({ai2_name}): {self.stats['ai2_wins']} ({self.stats['ai2_wins']/self.games*100:.1f}%)")
        # print(f"Draws: {self.stats['draws']} ({self.stats['draws']/self.games*100:.1f}%)")



        # Clean up pygame
        if self.display_game:
            pg.quit()

    def prepare_training_data(self):
        """
        Chỉ lấy dữ liệu từ người chiến thắng
        Trả về:
            X: các trạng thái bàn cờ khi người thắng thực hiện nước đi (7,6,1)
            y: các nước đi tương ứng của người thắng (one-hot vector 7 chiều)
        """
        X = []
        y = []
        history_games = self.history_games
        for game in history_games:
            # Xác định người chiến thắng
            winner = None
            if game.check_win(RED):
                winner = RED
            elif game.check_win(YELLOW):
                winner = YELLOW
            else:
                continue  # Bỏ qua nếu hòa
            
            count = 0
            # Chỉ lấy các nước đi của người thắng
            for state, move_valuated in game.history[winner]:
                state[state == 0] = 0
                X.append(state)
                max_value = max(move_valuated)
                one_hot = [1 if v == max_value else 0 for v in move_valuated]

                y.append(one_hot)
            
        return np.array(X), np.array(y)
# Chưa chỉnh lại cmd
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run AI vs AI Connect Four games')
    parser.add_argument('--ai1', type=int, choices=[0, 1, 2, 3, 4], default=3,
                        help='AI level for player 1 (Red) (default: 3 - Think Three)')
    parser.add_argument('--ai2', type=int, choices=[0, 1, 2, 3, 4], default=4,
                        help='AI level for player 2 (Yellow) (default: 4 - Monte Carlo Tree Search)')
    parser.add_argument('--nogui', action='store_true',
                        help='Run without graphical display')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between moves in seconds (default: 0.5)')
    parser.add_argument('--games', type=int, default=1,
                        help='Number of games to play (default: 1)')
    parser.add_argument('--timeout1', type=float, 
                        help='Timeout for AI 1 in seconds (default: based on AI level)')
    parser.add_argument('--timeout2', type=float,
                        help='Timeout for AI 2 in seconds (default: based on AI level)')
    parser.add_argument('--timeout', type=float,
                        help='Timeout for both AIs in seconds (overrides individual timeouts)')
    
    args = parser.parse_args()
    
    # Handle timeouts
    ai1_timeout = args.timeout1
    ai2_timeout = args.timeout2
    
    # Global timeout overrides individual timeouts
    if args.timeout is not None:
        ai1_timeout = args.timeout
        ai2_timeout = args.timeout
    
    # Set up the AI vs AI game
    # ai_vs_ai = MatchMaker(
    #     ai1_level=AIDifficulty(args.ai1),
    #     ai2_level=AIDifficulty(args.ai2),
    #     display_game=not args.nogui,
    #     delay=args.delay,
    #     games=args.games
    # )
    
    # Run the games
    ai_vs_ai.run()

if __name__ == '__main__':
    # main()

    # file_path = [os.path.abspath("DL/data/Train_RandomizeMinimax.npy"), 
    #              os.path.abspath("DL/data/Label_RandomizeMinimax.npy")]
    file_path = [None, None]

    # Set up the AI vs AI game
    ai_vs_ai = MatchMaker(
        player2=MinimaxAI2(depth=6),
        player1=MinimaxAI(),
        display_game=True,
        delay=0.5,
        games=1
    )

    os.system("cls" if os.name == "nt" else "clear")   
    # Run the games

    ai_vs_ai.run()
    # X, y = ai_vs_ai.prepare_training_data()
    # np.savez("DL/data/data_MO_MO_7.npz", X=X, y=y)

