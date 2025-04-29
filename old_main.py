import numpy as np
import random
import time
from numba import njit
COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
MODE_FLAT = 0
MODE_WEIGHTED = 1
MODE_WEIGHTED_FREE = 2
RANK_0 = 100000
RANK_1 = -100
RANK_2 = 10
RANK_3 = -5
RANK_4 = 5
RANK_5 = -2
WEIGHT_ONE = np.array([ 
    [RANK_0, RANK_1, RANK_2, RANK_5 , RANK_5, RANK_2, RANK_1, RANK_0],
    [RANK_1, RANK_2, RANK_4, RANK_3, RANK_3, RANK_4, RANK_2, RANK_1],
    [RANK_2, RANK_4, RANK_5, RANK_5, RANK_5, RANK_5, RANK_4, RANK_2],
    [RANK_5, RANK_3, RANK_5, RANK_5, RANK_5, RANK_5, RANK_3, RANK_5],
    [RANK_5, RANK_3, RANK_5, RANK_5, RANK_5, RANK_5, RANK_3,RANK_5],
    [RANK_2, RANK_4, RANK_5, RANK_5, RANK_5, RANK_5, RANK_4, RANK_2],
    [RANK_1, RANK_2, RANK_4, RANK_3, RANK_3, RANK_4, RANK_2, RANK_1],
    [RANK_0, RANK_1, RANK_2, RANK_5 , RANK_5, RANK_2, RANK_1, RANK_0],
])
random.seed(0)
class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
    def go(self, chessboard):
        self.candidate_list.clear()
        chessboard_as_black = self.adjust_board_as_black(chessboard)
        board_as_black = Board_As_Black(chessboard_as_black, self.chessboard_size)
        for available_child in board_as_black.get_child_board():
            self.candidate_list.append(available_child.last_move)
        if len(self.candidate_list) == 0:
            return self.candidate_list
        self.candidate_list.append(self.get_the_best_move(board_as_black))
        return self.candidate_list

    def get_the_best_move(self, board_as_black):
        # if self.count_empty_positions(board_as_black) >= 32:
        #     _,best_move = self.minimax(board_as_black, 3, MODE_WEIGHTED)
        # elif self.count_empty_positions(board_as_black) >= 16:
        #     _,best_move = self.minimax(board_as_black, 4, MODE_WEIGHTED_FREE)
        # else:
        #     _,best_move = self.minimax(board_as_black, 8, MODE_FLAT)
        _, best_move = self.minimax(board_as_black, 9, MODE_FLAT)
        return best_move
    
    def count_empty_positions(self, board_as_black):
        return len(board_as_black.get_empty_positions())

    def minimax(self, board_as_black, max_depth, mode):
        board_min = Board_As_Black(self.flip_the_board(board_as_black.chessboard), self.chessboard_size)
        def max_value(board_min, alpha, beta, max_depth):
            board_max = Board_As_Black(self.flip_the_board(board_min.chessboard), self.chessboard_size, board_min.parent, board_min.last_move, board_min.depth)
            if self.terminal_test(board_max):
                return self.utility_for_max(board_max, mode), None
            if board_max.depth >= max_depth:
                return self.utility_for_max(board_max, mode), None
            v, best_move = -np.inf, None
            if len(board_max.get_child_board()) != 0:
                child_boards = board_max.get_child_board()
                random.shuffle(child_boards)
                for child in child_boards:
                    min_choice , _ = min_value(child, alpha, beta, max_depth)
                    if min_choice > v:
                        v = min_choice
                        best_move = child.last_move
                    if v >= beta:
                        return v, best_move
                    alpha = max(alpha, v)
                return v, best_move
            else:
                return min_value(board_max, -np.inf, np.inf, max_depth)

        def min_value(board_max, alpha, beta, max_depth):
            board_min = Board_As_Black(self.flip_the_board(board_max.chessboard), self.chessboard_size, board_max.parent, board_max.last_move, board_max.depth)
            if self.terminal_test(board_min):
                return self.utility_for_min(board_min, mode), None
            if board_min.depth >= max_depth:
                return self.utility_for_min(board_min, mode), None
            v, best_move = +np.inf, None
            if len(board_min.get_child_board()) != 0:
                child_boards = board_min.get_child_board()
                random.shuffle(child_boards)
                for child in child_boards:
                    max_choice , _ = max_value(child, alpha, beta, max_depth)
                    if max_choice < v:
                        v = max_choice
                        best_move = child.last_move
                    if v <= alpha:
                        return v, best_move
                    beta = min(beta, v)
                return v, best_move
            else:
                return max_value(board_min, -np.inf, np.inf, max_depth)
            
        return max_value(board_min, -np.inf, np.inf, max_depth)
            
    def flip_the_board(self, chessboard):
        return -chessboard.copy()

    def adjust_board_as_black(self, chessboard):
        if self.color == COLOR_BLACK:
            return chessboard.copy()
        else:
            return -chessboard.copy()

    def terminal_test(self, board_as_black):
        if len(board_as_black.get_empty_positions()) == 0:
            return True
        else:
            if len(board_as_black.get_child_board()) == 0:
                fliped_board = Board_As_Black(self.flip_the_board(board_as_black.chessboard), self.chessboard_size, board_as_black.parent, board_as_black.last_move)
                if len(fliped_board.get_child_board()) == 0:
                    return True
        return False
    
    def utility_for_max(self, board_as_black, mode):
        if mode == MODE_FLAT:
            return np.sum(board_as_black.chessboard)
        elif mode == MODE_WEIGHTED:
            return np.sum(board_as_black.chessboard * WEIGHT_ONE)/(self.chessboard_size**2 - self.count_empty_positions(board_as_black)) + np.exp(0.6*np.sum(board_as_black.chessboard))
        elif mode == MODE_WEIGHTED_FREE:
            return np.sum(board_as_black.chessboard * WEIGHT_ONE)/(self.chessboard_size**2 - self.count_empty_positions(board_as_black)) - np.exp(0.6*len(board_as_black.get_child_board())) + np.exp(0.6*np.sum(board_as_black.chessboard))
    
    def utility_for_min(self, board_as_black, mode):
        if mode == MODE_FLAT:
            return -np.sum(board_as_black.chessboard)
        elif mode == MODE_WEIGHTED:
            return -np.sum(board_as_black.chessboard * WEIGHT_ONE)/(self.chessboard_size**2 - self.count_empty_positions(board_as_black)) - np.exp(0.4*np.sum(board_as_black.chessboard))
        elif mode == MODE_WEIGHTED_FREE:
            return -np.sum(board_as_black.chessboard * WEIGHT_ONE)/(self.chessboard_size**2 - self.count_empty_positions(board_as_black)) - np.exp(0.6*len(board_as_black.get_child_board())) - np.exp(0.4*np.sum(board_as_black.chessboard))

class Board_As_Black(object):
    def __init__(self, chessboard, chessboard_size, parent=None, last_move=None, depth=0):
        self.chessboard_size = chessboard_size
        self.chessboard = chessboard
        self.last_move = last_move
        self.child_board = []
        self.parent = parent
        self.depth = depth

    def get_child_board(self):
        targets = self.get_empty_positions()
        for target in targets:
            child_board = self.get_child_board_for_specific_move(target)
            if child_board is not None:
                self.child_board.append(child_board)
        return self.child_board 

    def get_child_board_for_specific_move(self, target):
        x_position = target[0]
        y_position = target[1]
        new_board = Board_As_Black(self.chessboard.copy(), self.chessboard_size, self, target, self.depth + 1)
        new_board.chessboard[x_position][y_position] = COLOR_BLACK
        directions = np.array([[-1,-1],[0,1],[1,0],[1,1],[0,-1],[-1,0],[1,-1],[-1,1]])
        whether_changed = False
        for direction in directions:
            change_in_x = direction[0]
            change_in_y = direction[1]
            new_x_position = x_position + change_in_x
            new_y_position = y_position + change_in_y
            if self.get_whether_out_of_range(new_x_position, new_y_position):
                continue
            elif new_board.chessboard[new_x_position][new_y_position] != COLOR_WHITE:
                continue
            else:
                while True:
                    new_x_position += change_in_x
                    new_y_position += change_in_y
                    if self.get_whether_out_of_range(new_x_position, new_y_position):
                        break
                    elif new_board.chessboard[new_x_position][new_y_position] == COLOR_BLACK:
                        whether_changed = True
                        while new_x_position != x_position or new_y_position != y_position:
                            new_x_position -= change_in_x
                            new_y_position -= change_in_y
                            new_board.chessboard[new_x_position][new_y_position] = COLOR_BLACK
                        break
                    elif new_board.chessboard[new_x_position][new_y_position] == COLOR_WHITE:
                        continue
                    else:
                        break
        if whether_changed:
            return new_board
        else:
            return None
        

    def get_whether_out_of_range(self, x_position, y_position):
        return x_position < 0 or x_position >= self.chessboard_size or y_position < 0 or y_position >= self.chessboard_size

    def get_empty_positions(self):
        targets =  np.where(self.chessboard == COLOR_NONE)
        return list(zip(targets[0], targets[1]))

import time

def tic():
    global start_time
    start_time = time.perf_counter()

def toc():
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    return elapsed_time

if __name__ == "__main__":
    tic()
    chessboard = np.array([
    [ 1,  0,  1,  0],
    [ 0, -1,  1,  0],
    [ 0,  1, -1,  0],
    [ 0,  0,  0,  0]
    ])
    ai = AI(4, COLOR_BLACK, 5)
    moves = ai.go(chessboard)
    print(moves)
    toc()
