import subprocess
import json
from typing import List, Optional

class Connect4Solver:
    def __init__(self):
        self.history = []  # Lưu lịch sử các nước đi (cột 0-6)
    
    def make_move(self, col: int) -> None:
        """Thêm nước đi vào lịch sử"""
        if not 0 <= col <= 6:
            raise ValueError("Cột phải từ 0-6")
        if self.history.count(col) >= 6:
            raise ValueError(f"Cột {col} đã đầy")
        self.history.append(col)
    
    def reset(self):
        self.history = []

    def get_optimal_move(self) -> Optional[int]:
        """
        Lấy nước đi tối ưu bằng curl
        Trả về None nếu có lỗi
        """
        if not self.history:
            return 3  # Ưu tiên cột giữa nếu bàn cờ trống
        
        try:
            # 1. Chuyển đổi lịch sử nước đi
            pos = ''.join(str(col + 1) for col in self.history)
            
            # 2. Gọi curl với timeout
            cmd = ['curl', '-s', f'https://connect4.gamesolver.org/solve?pos={pos}']
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # 3. Xử lý kết quả
            if result.returncode == 0:
                data = json.loads(result.stdout)
                scores = data.get("score", [])
                print(scores)
                
                if scores and len(scores) == 7:
                    # Tìm cột có điểm cao nhất và còn trống
                    valid_scores = [
                        (score, col) for col, score in enumerate(scores) 
                        if self.history.count(col) < 6
                    ]
                    if valid_scores:
                        center = 3
                        # best_col = max(valid_scores)[1]
                        best_col = max(
                            valid_scores,
                            key=lambda x: (x[0], x[1] % 2, -abs(x[1] - center))
                            # Ưu tiên: điểm cao → cột lẻ (1) → gần trung tâm
                        )[1]
                        return best_col
            
            return self.__fallback_move()
            
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            print(f"Lỗi khi gọi API: {type(e).__name__}")
            return self.__fallback_move()
        except Exception as e:
            print(f"Lỗi không xác định: {e}")
            return self.__fallback_move()

    def __fallback_move(self) -> int:
        """Thuật toán dự phòng khi API lỗi"""
        # Ưu tiên cột giữa và các cột còn trống
        priority_order = [3, 2, 4, 1, 5, 0, 6]
        for col in priority_order:
            if self.history.count(col) < 6:
                return col
        return 0  # Fallback cuối cùng