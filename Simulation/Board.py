### Tên file gốc:   AIconnect4.py

import numpy as np
import pickle
from copy import deepcopy

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Constant import RED, YELLOW, IDLE


class ConnectFourBoard:
    ###
    #
    #  Một giả lập sàn đấu, không bao hàm giao diện. 
    #  Tương đương tương tác trực tiếp tới ma trận biểu diễn trận đấu.
    #  Không bao gồm vẽ giao diện, không bao gồm AI xử lý.
    #
    #   Attributes:
    #       rows x columns  :   kích thước ma trận
    #       board           :   bàn cờ (??? RED/YELLOW là giá trị nào thì chịu)
    #       turn            :   lượt HIỆN TẠI do bên nào đi ( 1: red, -1: yellow)
    #       available       :   Vị trí sẽ được đánh vào tiếp theo của mỗi cột
    #       history         :   history[RED] sẽ lưu trữ lịch sử đấu của RED
    ###

    def __init__(self, shape:tuple[int, int] =(6, 7), first_to_move:int =RED, save_history:bool =False):
        """Initialize the Connect Four game."""
        # Board dimensions
        self.rows = shape[0]
        self.columns = shape[1]
        self.shape = shape
        
        # Game state
        self.create_board()
        self.turn = first_to_move
        self.first_to_move = first_to_move

        if save_history :
            # Each record contain [current_state, move_valuation]. Will helpl on training DQN model
            print('Has history')
            self.history:dict[int: list[tuple[np.array, np.array]]] \
                    = {RED:    [],
                       YELLOW: []}
        
    def append_history(self, current_state:np.array, move_valuate:np.array) :
        """Append history-stack of the game board, if it did save history"""

        if hasattr(self, 'history') :           # if this board was initilized with history

            self.history[self.turn].append((deepcopy(current_state) * self.turn, move_valuate))


    def create_board(self, initilize_state:np.array =None):
        """Create an empty game board."""

        if initilize_state is not None :
            self.board = deepcopy(initilize_state)
            self.shape = initilize_state.shape
            self.rows, self.columns = initilize_state.shape
        else :
            self.board = np.zeros(shape=(self.rows, self.columns)) 

        return self.board

    def reset_game(self, firstMoving=RED, initilize_state:np.array =None):
        """Reset the game to its initial state."""
        self.first_to_move = firstMoving
        self.turn = firstMoving
        
        self.create_board(initilize_state)


        if hasattr(self, 'history') :
            self.history[RED] = []
            self.history[YELLOW] = []
            
    def get_available(self) :
        """
        Danh sách vị trí hàng sẽ được đánh ở lượt tiếp theo của từng cột.
        Ví dụ: Cột 3 còn 3 vị trí trống thì: available[3] = 2
        """

        return ((self.rows - 1) - np.count_nonzero(self.board, axis=0))

    def get_available_columns(self) :
        """Trả về list các cột còn vị trí ô trống. Không ghi rõ trống tới hàng nào."""
        unordered = np.where(self.get_available() > -1)[0]
        mid = unordered.size // 2  # Xác định chỉ mục giữa
        ordered = [unordered[mid]]  # Bắt đầu với phần tử ở giữa
        
        left, right = mid - 1, mid + 1
        while left >= 0 or right < unordered.size:
            if right < unordered.size:
                ordered.append(unordered[right])
                right += 1
            if left >= 0:
                ordered.append(unordered[left])
                left -= 1
        
        return ordered

    def drop_piece(self, column:int, move_valuated:np.array=None):
        """Attempt to drop a piece in the specified column.
        
        Args:
            column: Column index to drop the piece
            turn  : Next move made by whom (Constant.RED or Constant.YELLOW only)
            
        Returns:
            bool: True if the piece was successfully dropped, False otherwise
        """
        
        row = self.rows - np.count_nonzero(self.board[:, column]) - 1
        if (row < 0) :
            return False
        else :
            if move_valuated is None :
                move_valuated = np.zeros((self.columns,))
                move_valuated[column] = 1

            self.append_history(self.board, move_valuated)

            self.board[row, column] = self.turn
            self.turn = -self.turn      
            return True

    def roll_back(self, destination_step:int, erase_all_history=False):
        """
        Roll to 'destination_step'.
        Give positive number, then come to exactly that turn.
        Give negative number, then it comeback abs(step_back).
        For example: 0 is init_state, while -1 is previous_state from now.

        Notice: 1 step here mean [1 RED turn, 1 YELLOW turn] a.k.a (1 cycle)  
        Notice: rollback will erase all history after the destination
        """
        if not hasattr(self, 'history') : 
            print('Sorry, this board didn\'t save history')
            return

        current_turn = self.turn
        other_turn = -current_turn

        self.history[current_turn] = self.history[current_turn][0:destination_step]

        if current_turn == self.first_to_move :
            if destination_step < -1 :
                self.history[other_turn] = self.history[other_turn][0:destination_step + 1]
            elif destination_step > 0 :
                self.history[other_turn] = self.history[other_turn][0:destination_step]
        else :
            if destination_step != -1 :
                self.history[other_turn] = self.history[other_turn][0:destination_step+1]

        if self.history[other_turn].size == 0 :
            self.create_board()
        else:
            self.create_board(self.history[other_turn].pop())

        if erase_all_history :
            self.history[RED] = []
            self.history[YELLOW] = []

    def check_win(self, turn:int=None, special_position:tuple[int, int]=None):
        """Check if the given piece has won on the board."""

        if special_position is not None :
            r = special_position[0]
            c = special_position[1]
            
            turn = self.board[r, c]
            if turn == IDLE :
                return False
            
        if turn is None:
            turn = self.turn

        def check_horizontal_win():
            """Check for a horizontal win."""
            if special_position is not None :
                r = special_position[0]
                for c in range(max(special_position[1] - 3, 0), min(special_position[1] + 1, self.columns - 3)):
                    if all(self.board[r][c + i] == turn for i in range(4)):
                        return True
                return False
            
            for r in range(self.rows):
                for c in range(self.columns - 3):
                    if all(self.board[r][c + i] == turn for i in range(4)):
                        return True
            return False
        
        def check_vertical_win():
            """Check for a vertical win."""
            if special_position is not None :
                c = special_position[1]
                for r in range(max(special_position[0] - 3, 0), min(special_position[0] + 1, self.rows - 3)):
                    if all(self.board[r + i][c] == turn for i in range(4)):
                        return True
                return False
        
            for r in range(self.rows - 3):
                for c in range(self.columns):
                    if all(self.board[r + i][c] == turn for i in range(4)):
                        return True
            return False
        
        def check_diagonal_down_win():
            """Check for a diagonal win (top-left to bottom-right)."""
            if special_position is not None :
                subtract = special_position[1] - special_position[0]
                for r in range(max(special_position[0] - 3, 0), min(special_position[0] + 1, self.rows - 3)):
                    c = r + subtract
                    if (r >= self.rows) :
                        break
                    if all(self.board[r + i][c + i] == turn for i in range(4)):
                        return True

            for r in range(self.rows - 3):
                for c in range(self.columns - 3):
                    if all(self.board[r + i][c + i] == turn for i in range(4)):
                        return True
            return False
        
        def check_diagonal_up_win():
            """Check for a diagonal win (bottom-left to top-right)."""
            if special_position is not None :
                sum = special_position[0] + special_position[1]
                for c in range(max(special_position[1] - 3, 0), min(special_position[1] + 1, self.columns - 3)) :
                    r = sum - c
                    if c >= self.columns:
                        break
                    if all(self.board[r - i][c + i] == turn for i in range(4)):
                        return True
                return False

            for r in range(3, self.rows):
                for c in range(self.columns - 3):
                    if all(self.board[r - i][c + i] == turn for i in range(4)):
                        return True
            return False

        return (check_horizontal_win() or 
                check_vertical_win() or 
                check_diagonal_down_win() or 
                check_diagonal_up_win())
    
    def is_full(self) :
        """Check if the board is full of piece. There no place to keep playing"""

        return np.count_nonzero(self.board) == self.board.size
    def copy(self) :
        clone = ConnectFourBoard(shape=self.shape, save_history=False)
        clone.reset_game(self.turn, self.board)
        return clone
    
    def export_history(self, color, train_file_path='data/DefaultTrainingSet.npy', label_file_path='data/DefaultLabelSet.npy') :
        """Append dữ liệu mới vào file .npy"""

        if len(self.history[color]) <= 4:       # bỏ qua việc export đối với trận đấu quá ngắn (do AI lỗi)
            return

        # Tách train_data và label_data từ danh sách các tuple
        new_train_data = np.squeeze(np.array([x[0] for x in self.history[color][2:]]))      # bỏ qua 2 turn đầu mỗi bên, vì nó ko hiệu quả lắm
        new_label_data = np.squeeze(np.array([x[1] for x in self.history[color][2:]]))

        if os.path.exists(train_file_path) and os.path.exists(label_file_path) :
            # Load lại dữ liệu cũ
            train_data = np.squeeze(np.load(train_file_path))
            label_data = np.squeeze(np.load(label_file_path))

            # Append dữ liệu mới
            new_train_data = np.concatenate((train_data, new_train_data), axis=0)
            new_label_data = np.concatenate((label_data, new_label_data), axis=0)

        np.save(train_file_path, new_train_data)
        np.save(label_file_path, new_label_data)


    def load_history_data(train_file_path='data/DefaultTrainingSet.npy', label_file_path='data/DefaultLabelSet.npy') :
        """Khuyến khích tự viết lại hàm load, không thì bay RAM"""

        train_data = None
        label_data = None

        if os.path.exists(train_file_path) and os.path.exists(label_file_path):
                # Load lại dữ liệu cũ
            train_data = np.load(train_file_path)
            label_data = np.load(label_file_path)
        
        return train_data, label_data
