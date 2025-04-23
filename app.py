from fastapi import FastAPI, HTTPException
import random
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import sys
import os
import time
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AI_AlphaGo.minimaxVsABPrunning import MinimaxAI
from AI_AlphaGo.minimaxAndRandom import MinimaxAI2
from AI_AlphaGo.Deep import Deep
from Simulation.Board import ConnectFourBoard
from AI_AlphaGo.hack.we import Connect4Solver
from AI_AlphaGo.minimaxDepthInc import minimaxDepthInc

app = FastAPI()
connect4 = Connect4Solver()
gm = ConnectFourBoard()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]
    is_new_game: bool

class AIResponse(BaseModel):
    move: int

def solution(game_state):
    board = game_state.board
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 2:
                board[i][j] = -1
    return board

def reverse_turn(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] in [-1, 1]:
                board[i][j] *= -1
    return board

def find_new_move_column(board_old, board_new):
    for col in range(7):
        old_col = board_old[:, col]
        new_col = board_new[:, col]
        if not np.array_equal(old_col, new_col):
            # Tìm dòng đầu tiên mà khác nhau, theo chiều từ dưới lên
            for row in range(5, -1, -1):
                if old_col[row] != new_col[row]:
                    return col
    return None

from typing import List

def reset_game(board: List[List[int]]):
    # Flatten board thành list 1 chiều
    flat_board = [cell for row in board for cell in row]

    count_ones = flat_board.count(1)
    count_nonzeros = sum(1 for x in flat_board if x != 0)

    if count_nonzeros == 0 or (count_ones == 1 and count_nonzeros == 1):
        global gm, connect4
        gm = ConnectFourBoard()
        connect4 = Connect4Solver()

from AI_AlphaGo.minimaxVsABPrunning import MinimaxAI
from AI_AlphaGo.minimaxAndMCTS import minimaxAndMcts
from Simulation.Board import ConnectFourBoard


@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    st = time.time()
    # print("hihi")
    try:
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")
        # print(game_state.board)
        # game = ConnectFourBoard()
        # game.board = np.array(solution(game_state))
        # myAI = MinimaxAI()
        # use_deep = False
        # if game_state.current_player == 2:
        #     if use_deep:
        #         myAI.set_color(-1)
        #         game.turn = 1
        #     else:
        #         # game.board = reversed(game.board.copy())
        #         myAI.set_color(-1)
        #         game.turn = -1
        # else:
        #     myAI.set_color(1)
        #     game.turn = 1
        # selected_move = myAI.get_move(game)[0]
        ## ///////////////
        board = np.array(solution(game_state))
        reset_game(board)
        global gm
        col = find_new_move_column(np.array(gm.board), board)
        selected_move = 3
        if col == None:
            print("new game")
            selected_move = connect4.get_optimal_move()
        else:
            print("diff")
            connect4.make_move(col)
            gm.drop_piece(col)
            selected_move = connect4.get_optimal_move()
        connect4.make_move(selected_move)
        gm.drop_piece(selected_move)
        print(selected_move)
        print(time.time() - st)
        return AIResponse(move=selected_move)
    except Exception as e:
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[1])
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
