import numpy as np
import random
import time
import math
from numba import njit

# 常量定义
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
    [RANK_0, RANK_1, RANK_2, RANK_5, RANK_5, RANK_2, RANK_1, RANK_0],
    [RANK_1, RANK_2, RANK_4, RANK_3, RANK_3, RANK_4, RANK_2, RANK_1],
    [RANK_2, RANK_4, RANK_5, RANK_5, RANK_5, RANK_5, RANK_4, RANK_2],
    [RANK_5, RANK_3, RANK_5, RANK_5, RANK_5, RANK_5, RANK_3, RANK_5],
    [RANK_5, RANK_3, RANK_5, RANK_5, RANK_5, RANK_5, RANK_3, RANK_5],
    [RANK_2, RANK_4, RANK_5, RANK_5, RANK_5, RANK_5, RANK_4, RANK_2],
    [RANK_1, RANK_2, RANK_4, RANK_3, RANK_3, RANK_4, RANK_2, RANK_1],
    [RANK_0, RANK_1, RANK_2, RANK_5, RANK_5, RANK_2, RANK_1, RANK_0],
])

random.seed(0)

class MCTSNode:
    def __init__(self, board, parent=None, last_move=None, ai_color=COLOR_BLACK):
        # 统一转换为黑棋视角存储
        self.board = Board_As_Black(
            board.chessboard if ai_color == COLOR_BLACK else -board.chessboard,
            board.chessboard_size,
            parent,
            last_move,
            board.depth
        )
        self.parent = parent
        self.last_move = last_move
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = None
        self.ai_color = ai_color  

    def get_untried_moves(self):
        if self.untried_moves is None:
            self.untried_moves = self.board.get_child_board()
        return self.untried_moves

    def select_child(self, exploration=1.414):
        selected = None
        best_value = -float('inf')
        
        for child in self.children:
            if child.visits == 0:
                uct_value = float('inf')
            else:
                exploitation = child.wins / child.visits
                exploration_term = exploration * math.sqrt(math.log(self.visits) / child.visits)
                uct_value = exploitation + exploration_term
            
            if uct_value > best_value:
                best_value = uct_value
                selected = child
        return selected

    def expand(self):
        untried_moves = self.get_untried_moves()
        if not untried_moves:
            return None
            
        move = random.choice(untried_moves)
        self.untried_moves.remove(move)
        
        child_node = MCTSNode(move, parent=self, last_move=move.last_move, ai_color=-self.ai_color)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

class Board_As_Black:
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
        x, y = target
        new_board = Board_As_Black(self.chessboard.copy(), self.chessboard_size, self, target, self.depth + 1)
        new_board.chessboard[x][y] = COLOR_BLACK
        
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        flipped = False
        
        for dx, dy in directions:
            tx, ty = x + dx, y + dy
            if not (0 <= tx < self.chessboard_size and 0 <= ty < self.chessboard_size):
                continue
            if new_board.chessboard[tx][ty] != COLOR_WHITE:
                continue
                
            path = []
            while True:
                tx += dx
                ty += dy
                if not (0 <= tx < self.chessboard_size and 0 <= ty < self.chessboard_size):
                    break
                if new_board.chessboard[tx][ty] == COLOR_BLACK:
                    flipped = True
                    for px, py in path:
                        new_board.chessboard[px][py] = COLOR_BLACK
                    break
                elif new_board.chessboard[tx][ty] == COLOR_WHITE:
                    path.append((tx, ty))
                else:
                    break
        
        return new_board if flipped else None

    def get_empty_positions(self):
        return list(zip(*np.where(self.chessboard == COLOR_NONE)))

class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.mcts_time_limit = 4.6  # MCTS搜索时间限制(秒)

    def go(self, chessboard):
        self.candidate_list.clear()
        chessboard_as_black = self.adjust_board_as_black(chessboard)
        board_as_black = Board_As_Black(chessboard_as_black, self.chessboard_size)
        
        # 获取所有合法移动
        for available_child in board_as_black.get_child_board():
            self.candidate_list.append(available_child.last_move)
        if not self.candidate_list:
            return self.candidate_list
            
        # 根据游戏阶段选择策略
        empty_count = len(board_as_black.get_empty_positions())
        if empty_count >= 35:  # 早期使用MCTS
            best_move = self.mcts_search(board_as_black)
        elif empty_count >= 20:  # 中期使用minimax
            _, best_move = self.minimax(board_as_black, 4, MODE_WEIGHTED_FREE)
        else:  # 后期使用更深的minimax
            _, best_move = self.minimax(board_as_black, 8, MODE_FLAT)
        
        if best_move:
            self.candidate_list.append(best_move)
        return self.candidate_list

    def mcts_search(self, root_board, iterations=1000):
        root_node = MCTSNode(root_board, ai_color=self.color)
        
        start_time = time.time()
        iteration = 0
        
        while iteration < iterations and (time.time() - start_time) < self.mcts_time_limit:
            node = root_node
            board = root_board
            
            # 选择阶段
            while node.children and not node.get_untried_moves():
                node = node.select_child()
                board = node.board
            
            # 扩展阶段
            if node.get_untried_moves():
                move = random.choice(node.get_untried_moves())
                board = move
                node = node.expand()
            
            # 模拟阶段
            result = self.simulate(board)
            
            # 回溯阶段
            while node is not None:
                node.update(result)
                node = node.parent
            
            iteration += 1
        
        # 返回访问次数最多的子节点
        if root_node.children:
            return max(root_node.children, key=lambda c: c.visits).last_move
        return None

    def simulate(self, board):
        """
        从当前节点开始随机模拟游戏直到终局
        返回: 
            1  : 当前AI胜利
            0  : 平局
            -1 : 当前AI失败
        """
        # 初始化模拟状态（注意：board已是当前玩家视角）
        current_board = board
        current_color = self.ai_color  # 从当前节点的玩家颜色开始
        
        while True:
            # 1. 检查游戏是否结束（棋盘已满）
            empty_positions = current_board.get_empty_positions()
            if not empty_positions:
                break
            
            # 2. 获取当前玩家的合法移动
            moves = current_board.get_child_board()
            
            # 3. 处理无合法移动的情况
            if not moves:
                # 3.1 尝试切换玩家
                flipped_board = Board_As_Black(
                    self.flip_the_board(current_board.chessboard),
                    current_board.chessboard_size
                )
                opponent_moves = flipped_board.get_child_board()
                
                # 3.2 双方都无合法移动时结束游戏
                if not opponent_moves:
                    break
                    
                # 3.3 切换玩家继续
                current_color = -current_color
                current_board = flipped_board
                continue
            
            # 4. 随机选择一个合法移动
            move = random.choice(moves)
            current_board = move
            current_color = -current_color  # 关键：落子后切换玩家
        
        # 5. 计算最终得分（基于黑棋视角）
        final_score = np.sum(current_board.chessboard)  # 黑棋: -1, 白棋: 1
        
        # 6. 根据AI执子颜色修正结果
        if self.ai_color == COLOR_BLACK:
            # AI执黑时：final_score>0 表示黑棋多（AI胜）
            return 1 if final_score > 0 else 0 if final_score == 0 else -1
        else:
            # AI执白时：final_score<0 表示白棋多（AI胜）
            return 1 if final_score < 0 else 0 if final_score == 0 else -1
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
# 测试代码
def main():
    # 初始化8x8棋盘
    initial_board = np.zeros((8, 8), dtype=int)
    initial_board[3][3] = initial_board[4][4] = COLOR_WHITE
    initial_board[3][4] = initial_board[4][3] = COLOR_BLACK

    # 测试1：初始局面测试
    print("测试1：初始局面")
    print_board(initial_board)
    test_ai(initial_board, COLOR_BLACK)  # 测试黑棋AI
    test_ai(initial_board, COLOR_WHITE)  # 测试白棋AI

    # 测试2：中局局面测试
    print("\n测试2：中局局面")
    mid_game = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1,-1,-1, 1, 0, 0],
        [0, 0,-1,-1,-1, 0, 0, 0],
        [0, 0, 0,-1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])
    print_board(mid_game)
    test_ai(mid_game, COLOR_BLACK)
    test_ai(mid_game, COLOR_WHITE)

    # 测试3：边缘局面测试
    print("\n测试3：边缘局面")
    edge_case = np.array([
        [-1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0,-1]
    ])
    print_board(edge_case)
    test_ai(edge_case, COLOR_BLACK)

def print_board(board):
    """打印8x8棋盘"""
    print("   a b c d e f g h")
    for i in range(8):
        print(f"{i+1} ", end="")
        for j in range(8):
            if board[i][j] == COLOR_BLACK:
                print("⚫", end="")
            elif board[i][j] == COLOR_WHITE:
                print("⚪", end="")
            else:
                print("⬜", end="")
        print()

def test_ai(board, ai_color):
    """测试AI在给定局面下的决策"""
    print(f"\nAI执{'黑' if ai_color == COLOR_BLACK else '白'}棋测试:")
    ai = AI(8, ai_color, 5)
    
    start_time = time.time()
    move = ai.go(board.copy())
    elapsed = time.time() - start_time
    
    if move and len(move) > 0:
        last_move = move[-1] if isinstance(move, list) else move
        print(f"推荐落子位置: ({last_move[0]+1}, {chr(last_move[1]+97)})")
    else:
        print("无合法移动")
    print(f"决策时间: {elapsed:.3f}秒")
    print("候选移动:", [f"({m[0]+1},{chr(m[1]+97)})" for m in move[:-1]] if move else "无")

if __name__ == "__main__":
    main()
