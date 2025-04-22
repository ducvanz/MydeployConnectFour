from Simulation.Board import ConnectFourBoard
from AI_AlphaGo.MCTS import simulate, expand
from Constant import RED, YELLOW, IDLE 
from copy import deepcopy
import random
import numpy as np
import math
import time
from math import sqrt, log
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

class minimaxAndMcts:
    def __init__(self, color=RED, timeout=5, exploration_factor=1.5, randomness=0.2):
        """
        Phiên bản nâng cao của Minimax + MCTS với các cải tiến:
        - Opening book cho các nước đi đầu
        - Pattern recognition
        - Progressive widening
        - Adaptive simulation
        - Enhanced heuristic evaluation
        """
        self.name = "EnhancedMinimaxMCTS"
        self.color = color
        self.max_time = timeout
        self.exploration_factor = exploration_factor
        self.randomness = randomness
        self.max_simulations = 1000
        self.min_simulations = 100
        self.rng = np.random.RandomState()
        self.stats = defaultdict(lambda: {'count': 0, 'wins': 0, 'losses': 0, 'depth': 0})
        
        # Opening book cho 3 nước đi đầu
        self.opening_book = {
            0: [3, 2, 4, 1, 5, 0, 6],  # Ưu tiên cột trung tâm trước
            1: [3, 2, 4, 1, 5, 0, 6],
            2: [3, 2, 4, 1, 5, 0, 6]
        }
        
        # Pattern database cho các thế cờ quan trọng
        self.patterns = {
            'winning': 100000,
            'block_win': 50000,
            'double_threat': 10000,
            'center_control': 50,
            'potential_threats': 200
        }
        
        # Adaptive parameters
        self.simulation_depth = 5  # Độ sâu mặc định
        self.adaptive_weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # Trọng số giảm dần theo độ sâu

    def set_color(self, color: int):
        self.color = color

    def seed(self, seed_value):
        self.rng = np.random.RandomState(seed_value)
        random.seed(seed_value)

    def get_move(self, game: ConnectFourBoard):
        # if np.random.randint() % 5 == 0:
        #     return np.random.choice(game.get_available_columns())
        start_time = time.time()
        self.stats.clear()
        
        # Kiểm tra opening book
        move_count = sum(1 for row in game.board for cell in row if cell != IDLE)
        if move_count < 3:
            available = game.get_available_columns()
            for move in self.opening_book[move_count]:
                if move in available:
                    return move, 0
        
        available_moves = self.get_ordered_moves(game)
        
        if len(available_moves) == 1:
            return available_moves[0], 0

        best_move = None
        best_value = -math.inf if self.color == RED else math.inf
        
        # Adaptive depth based on game phase
        self.adjust_simulation_depth(game)
        
        for move in available_moves:
            if time.time() - start_time > self.max_time * 0.9:
                break
                
            temp_game = deepcopy(game)
            temp_game.drop_piece(move)
            
            # Kiểm tra chiến thắng ngay lập tức
            if temp_game.check_win(self.color):
                return move, math.inf if self.color == RED else -math.inf
            
            # Kiểm tra ngăn chặn chiến thắng của đối thủ
            opponent_color = YELLOW if self.color == RED else RED
            if temp_game.check_win(opponent_color):
                continue  # Tránh nước đi này
                
            # Đánh giá nước đi với MCTS nâng cao
            move_value = self.enhanced_parallel_mcts(temp_game)
            
            # Thêm heuristic evaluation
            heuristic_val = self.enhanced_heuristic_evaluation(temp_game)
            move_value = 0.7 * move_value + 0.3 * heuristic_val
            
            # Cập nhật nước đi tốt nhất
            if (self.color == RED and move_value > best_value) or \
               (self.color == YELLOW and move_value < best_value) or \
               (best_move is None):
                best_value = move_value
                best_move = move

        # Progressive widening - mở rộng tìm kiếm khi có thời gian
        if time.time() - start_time < self.max_time * 0.5 and len(available_moves) > 1:
            second_best = self.find_alternative_move(game, best_move)
            if second_best is not None:
                temp_game = deepcopy(game)
                temp_game.drop_piece(second_best)
                second_value = self.enhanced_parallel_mcts(temp_game)
                
                # Đôi khi chọn nước đi thứ 2 để đa dạng hóa
                if abs(second_value - best_value) < 0.1 and self.rng.random() < 0.3:
                    best_move = second_best

        return best_move if best_move is not None else available_moves[0], best_value

    def get_ordered_moves(self, game):
        """Sắp xếp các nước đi theo heuristic + ngẫu nhiên"""
        moves = game.get_available_columns()
        center = game.columns // 2
        
        # Ưu tiên trung tâm nhưng thêm nhiễu ngẫu nhiên
        return sorted(moves, 
                    key=lambda x: abs(x - center) + self.rng.uniform(-0.5, 0.5))

    def adjust_simulation_depth(self, game):
        """Điều chỉnh độ sâu dựa trên giai đoạn trò chơi"""
        empty_cells = sum(1 for row in game.board for cell in row if cell == IDLE)
        total_cells = game.rows * game.columns
        
        if empty_cells > total_cells * 0.7:  # Giai đoạn đầu
            self.simulation_depth = 4
        elif empty_cells > total_cells * 0.3:  # Giai đoạn giữa
            self.simulation_depth = 6
        else:  # Giai đoạn cuối
            self.simulation_depth = 8

    def enhanced_parallel_mcts(self, game_state):
        """MCTS nâng cao với adaptive simulations"""
        start_time = time.time()
        local_stats = defaultdict(lambda: {'count': 0, 'wins': 0, 'losses': 0, 'depth': 0})
        
        with ThreadPoolExecutor() as executor:
            futures = []
            simulations = min(
                self.max_simulations,
                max(self.min_simulations, int(200 / (1 + len(game_state.get_available_columns())))))
            for _ in range(simulations):
                if time.time() - start_time > self.max_time * 0.8:
                    break
                    
                futures.append(executor.submit(
                    self.run_enhanced_simulation,
                    deepcopy(game_state)
                ))
            
            for future in as_completed(futures):
                try:
                    move, result, depth = future.result()
                    if move is not None:
                        local_stats[move]['count'] += 1
                        if result == game_state.turn:
                            local_stats[move]['wins'] += 1
                        elif result != IDLE:
                            local_stats[move]['losses'] += 1
                        local_stats[move]['depth'] = max(local_stats[move]['depth'], depth)
                except Exception as e:
                    print(f"Simulation error: {e}")

        # Tính giá trị trung bình có trọng số theo độ sâu
        move_values = []
        for move in local_stats:
            if local_stats[move]['count'] > 0:
                base_value = (local_stats[move]['wins'] - local_stats[move]['losses']) / local_stats[move]['count']
                depth_weight = 1.0 + (local_stats[move]['depth'] / self.simulation_depth) * 0.5
                move_values.append(base_value * depth_weight * self.rng.uniform(0.95, 1.05))
        
        return np.mean(move_values) if move_values else 0

    def run_enhanced_simulation(self, game_state):
        """Mô phỏng nâng cao với pattern matching và adaptive depth"""
        current_state = deepcopy(game_state)
        move_sequence = []
        depth = 0
        
        while depth < self.simulation_depth:
            # Kiểm tra chiến thắng trước khi mô phỏng
            if current_state.check_win(RED):
                if len(move_sequence) > 0:
                    return move_sequence[0], RED, depth
                return None, RED, depth
            if current_state.check_win(YELLOW):
                if len(move_sequence) > 0:
                    return move_sequence[0], YELLOW, depth
                return None, YELLOW, depth
                
            # Pattern matching - phát hiện các thế cờ quan trọng
            pattern_result = self.detect_special_patterns(current_state)
            if pattern_result is not None:
                best_move, score = pattern_result
                if len(move_sequence) == 0:
                    move_sequence.append(best_move)
                return move_sequence[0], score, depth
                
            # Chọn nước đi với progressive widening
            legal_moves = current_state.get_available_columns()
            if not legal_moves:
                return (move_sequence[0] if len(move_sequence) > 0 else None), IDLE, depth
                
            # Adaptive move selection
            if depth < 2:  # Ở độ sâu nông, xem xét nhiều nước đi hơn
                considered_moves = legal_moves
            else:
                considered_moves = random.sample(legal_moves, min(3, len(legal_moves)))
                
            best_move = max(considered_moves, key=lambda m: self.uct_value(current_state, m))
            
            # Thỉnh thoảng chọn ngẫu nhiên để khám phá
            if self.rng.random() < 0.1:
                best_move = self.rng.choice(legal_moves)
                
            if len(move_sequence) == 0:
                move_sequence.append(best_move)
                
            current_state.drop_piece(best_move)
            depth += 1
            
        # Mô phỏng phần còn lại
        result = simulate(current_state)
        return (move_sequence[0] if len(move_sequence) > 0 else None), result, depth

    def detect_special_patterns(self, game_state):
        """Phát hiện các thế cờ đặc biệt quan trọng"""
        # Kiểm tra chiến thắng trong 1 nước
        for move in game_state.get_available_columns():
            temp_game = deepcopy(game_state)
            temp_game.drop_piece(move)
            if temp_game.check_win(self.color):
                return move, math.inf if self.color == RED else -math.inf
                
        # Kiểm tra ngăn chặn chiến thắng đối thủ
        opponent_color = YELLOW if self.color == RED else RED
        for move in game_state.get_available_columns():
            temp_game = deepcopy(game_state)
            temp_game.drop_piece(move)
            if temp_game.check_win(opponent_color):
                return move, -math.inf if self.color == RED else math.inf
                
        # Kiểm tra double threat (tạo 2 cách chiến thắng)
        threats = []
        for move in game_state.get_available_columns():
            temp_game = deepcopy(game_state)
            temp_game.drop_piece(move)
            win_count = 0
            for next_move in temp_game.get_available_columns():
                next_temp = deepcopy(temp_game)
                next_temp.drop_piece(next_move)
                if next_temp.check_win(self.color):
                    win_count += 1
                    if win_count >= 2:
                        threats.append(move)
                        break
                        
        if threats:
            best_threat = max(threats, key=lambda m: self.patterns['double_threat'])
            return best_threat, self.patterns['double_threat'] * (1 if self.color == RED else -1)
            
        return None

    def enhanced_heuristic_evaluation(self, game):
        """Đánh giá heuristic nâng cao với pattern recognition"""
        if game.check_win(RED):
            return self.patterns['winning'] * (1 if self.color == RED else -1)
        if game.check_win(YELLOW):
            return self.patterns['winning'] * (1 if self.color == YELLOW else -1)
            
        score = 0
        
        # Đếm số lượng các pattern quan trọng
        red_score, yellow_score = self.count_patterns(game)
        
        if self.color == RED:
            score += red_score - yellow_score * 1.2  # Phạt nặng hơn các mối đe dọa
        else:
            score += yellow_score - red_score * 1.2
            
        # Thêm điểm kiểm soát trung tâm
        center_col = game.columns // 2
        for row in range(game.rows):
            if game.board[row][center_col] == self.color:
                score += self.patterns['center_control']
            elif game.board[row][center_col] != IDLE:
                score -= self.patterns['center_control'] * 0.5
                
        return score

    def count_patterns(self, game):
        """Đếm các pattern quan trọng trên bàn cờ"""
        red_score = 0
        yellow_score = 0
        
        # Kiểm tra tất cả các hàng, cột, đường chéo
        for row in range(game.rows):
            for col in range(game.columns):
                if game.board[row][col] == RED:
                    red_score += self.evaluate_position(game, row, col, RED)
                elif game.board[row][col] == YELLOW:
                    yellow_score += self.evaluate_position(game, row, col, YELLOW)
                    
        return red_score, yellow_score

    def evaluate_position(self, game, row, col, color):
        """Đánh giá vị trí với các pattern quan trọng"""
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            line_length = 1
            empty_ends = 0
            
            # Kiểm tra một hướng
            r, c = row + dr, col + dc
            while 0 <= r < game.rows and 0 <= c < game.columns:
                if game.board[r][c] == color:
                    line_length += 1
                elif game.board[r][c] == IDLE:
                    empty_ends += 1
                    break
                else:
                    break
                r += dr
                c += dc
                
            # Kiểm tra hướng ngược lại
            r, c = row - dr, col - dc
            while 0 <= r < game.rows and 0 <= c < game.columns:
                if game.board[r][c] == color:
                    line_length += 1
                elif game.board[r][c] == IDLE:
                    empty_ends += 1
                    break
                else:
                    break
                r -= dr
                c -= dc
                
            # Đánh giá pattern
            if line_length >= 4:
                score += self.patterns['winning']
            elif line_length == 3 and empty_ends >= 1:
                score += self.patterns['double_threat'] * 0.6
            elif line_length == 2 and empty_ends >= 2:
                score += self.patterns['potential_threats']
                
        return score

    def find_alternative_move(self, game, best_move):
        """Tìm nước đi thay thế tốt thứ 2"""
        available = game.get_available_columns()
        if len(available) < 2:
            return None
            
        alternatives = [m for m in available if m != best_move]
        if not alternatives:
            return None
            
        # Đánh giá nhanh bằng heuristic
        best_alt = None
        best_score = -math.inf if self.color == RED else math.inf
        
        for move in alternatives:
            temp_game = deepcopy(game)
            temp_game.drop_piece(move)
            score = self.enhanced_heuristic_evaluation(temp_game)
            
            if (self.color == RED and score > best_score) or \
               (self.color == YELLOW and score < best_score):
                best_score = score
                best_alt = move
                
        return best_alt
    def uct_value(self, game_state, move):
        stats = self.stats[move]
        total_simulations = sum(self.stats[m]['count'] for m in game_state.get_available_columns())
        
        # Nếu nước đi chưa được khám phá, trả về giá trị vô cùng
        if stats['count'] == 0:
            return float('inf')
        
        # Tính tỷ lệ thắng cơ bản (đã điều chỉnh cho màu quân)
        win_rate = stats['wins'] / stats['count']
        if self.color == YELLOW:
            win_rate = 1 - win_rate
        
        # Tính hệ số khám phá với điều chỉnh động
        exploration_weight = self.exploration_factor * (1 + 0.5 * math.sin(time.time()/10))  # Biến đổi theo thời gian
        exploration = exploration_weight * sqrt(log(total_simulations + 1) / (stats['count'] + 1))
        
        # Thêm bonus theo độ sâu trung bình
        depth_bonus = 0
        if stats['count'] > 0:
            depth_bonus = 0.1 * (stats['depth'] / self.simulation_depth)
        
        # Thêm đánh giá heuristic
        heuristic_bonus = 0
        if hasattr(self, 'enhanced_heuristic_evaluation'):
            temp_game = deepcopy(game_state)
            temp_game.drop_piece(move)
            heuristic_val = self.enhanced_heuristic_evaluation(temp_game)
            heuristic_bonus = 0.05 * np.tanh(heuristic_val / 10000)  # Chuẩn hóa về [-0.05, 0.05]
        
        # Giá trị UCT tổng hợp
        value = win_rate + exploration + depth_bonus + heuristic_bonus
        
        # Thêm nhiễu nhỏ để tránh hòa nhau
        value += self.rng.uniform(-0.0001, 0.0001)
        
        return value