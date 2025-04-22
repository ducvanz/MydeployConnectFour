import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Simulation.Board import ConnectFourBoard
from AI_AlphaGo.minimaxVsABPrunning import MinimaxAI, DEFAULT_WEIGHT
from Constant import RED, YELLOW, IDLE 

thesh_hold = [3/42, 7/42, 11/42, 17/42, 23/42, 27/42, 29/42]
### The idea is
### Start from depth=2, after each thresh_hold increase by 1
### Except for thresh_hold[0], it come from 2 go up to 4 immediately (the minimum to see the twin move like _**_)
### After thresh_hold[-1], it auto increase from 9 upto 19

class minimaxDepthInc :
    def __init__ (self, module=MinimaxAI) :
        self.module = module
        self.org_name = 'DepthInc '
        self.name = self.org_name + module().name
        self.color = IDLE

    def set_color(self, color) :
        self.color = color

    def suite_depth(game:ConnectFourBoard) :
        fill_percent = np.sum(game.board != 0) / 42
        level = 0
        while True :
            if level >= len(thesh_hold) :
                break
            if fill_percent <= thesh_hold[level] :
                break
            level += 1

        if level >= len(thesh_hold):
            return 5 + level + round((fill_percent - thesh_hold[-1]) * 42)
        elif level == 0 :
            return 2
        else :
            return 3 + level

    def get_move(self, game:ConnectFourBoard) :
        depth = minimaxDepthInc.suite_depth(game)
        
        self.name = self.org_name + f" depth={depth}"

        runner = self.module(depth=depth, color=self.color)
        return runner.get_move(game)