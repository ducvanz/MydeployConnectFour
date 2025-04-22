import numpy as np
import pygame as pg

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Simulation.Board import ConnectFourBoard
from Constant import RED, YELLOW, IDLE, WIDTH

class Hugeman :
    def __init__(self, screen_width=WIDTH, color=RED, timeout=None) :
        self.name = 'Hugeman player'
        self.width = screen_width
        self.color = color

    def set_color(self, color) :
        self.color = color

    def set_screen_width(self, screen_width) :
        self.width = screen_width

    def get_move(self, game:ConnectFourBoard) :
        square = self.width // game.columns

        while True :
            for event in pg.event.get():     
                if event.type == pg.QUIT:
                    return None, None
                elif event.type == pg.MOUSEBUTTONDOWN:
                    pos_x = event.pos[0]
                    col = pos_x // square
                    return col, None