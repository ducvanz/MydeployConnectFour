import subprocess
import json
from typing import List, Optional
import requests

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
        if not self.history:
            return 3

        try:
            pos = ''.join(str(col + 1) for col in self.history)
            url = f"https://connect4.gamesolver.org/solve?pos={pos}"
            headers = {
                "User-Agent": "Mozilla/5.0"  # giả lập browser
            }

            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                data = response.json()
                scores = data.get("score", [])
                print(scores)

                if scores and len(scores) == 7:
                    valid_scores = [
                        (score, col) for col, score in enumerate(scores)
                        if self.history.count(col) < 6
                    ]
                    if valid_scores:
                        center = 3
                        best_col = max(
                            valid_scores,
                            key=lambda x: (x[0], x[1] % 2, -abs(x[1] - center))
                        )[1]
                        return best_col
            return self.__fallback_move()

        except (requests.Timeout, requests.JSONDecodeError) as e:
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
