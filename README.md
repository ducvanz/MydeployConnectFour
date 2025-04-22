# ConnectFour
## Giai đoạn 1: xây dựng giao diện thử nghiệm - done
## Giai đoạn 2: Kết hợp AI sử dụng thuật toán minimax - cắt tỉa alpha, beta mà MCTS để tối ưu
- AI chạy được với giao diện thử nghiệm
- Thống kê lại thời gian chạy với từng độ sâu
- người thực hiện: Thuyết, Đức - (chính)| Văn, Núi - Hỗ trợ

### AI vs AI
- Chức năng cho phép hai AI chơi đối đầu với nhau
- Hỗ trợ 4 cấp độ AI:
  - Level 1: Think One - Tìm nước đi thắng ngay lập tức
  - Level 2: Think Two - Tìm nước đi thắng và chặn đối phương
  - Level 3: Think Three - Nhìn trước 3 nước đi
  - Level 4: Monte Carlo Tree Search - Sử dụng thuật toán MCTS

### Cách sử dụng
Chạy file `ai_vs_ai.py` với các tham số sau:
- `--ai1`: Cấp độ AI đỏ (1-4, mặc định: 3)
- `--ai2`: Cấp độ AI vàng (1-4, mặc định: 4)
- `--nogui`: Chạy không có giao diện đồ họa
- `--delay`: Độ trễ giữa các nước đi (mặc định: 0.5 giây)
- `--games`: Số trận đấu để chạy (mặc định: 1)
- `--timeout`: Thời gian tối đa cho mỗi AI suy nghĩ (tính bằng giây)
- `--timeout1`: Thời gian tối đa cho AI đỏ
- `--timeout2`: Thời gian tối đa cho AI vàng