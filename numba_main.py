import numpy as np
import random
import time
import math
from numba import njit

# 常量定义
COLOR_BLACK = -1
COLOR_WHITE = 1
BLACK_WIN = -1
WHITE_WIN = 1
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
WIN_POSITION = [(0,1), (1, 0), (6, 0), (7, 1), (0, 6), (1, 7), (6, 7), (7, 6)]
LOSE_POSITION = [(0,0), (7, 0), (0, 7), (7, 7)]
UNHAPPY_POSITION = [(2, 1), (1, 2), (5, 1), (6, 2), (1, 5), (2, 6), (5, 6), (6, 5),]
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
        if self.ai_color is COLOR_BLACK:
            for new_move in untried_moves:
                if new_move.last_move in WIN_POSITION:
                    move = new_move
                    break
            if move.last_move in LOSE_POSITION:
                move = random.choice(untried_moves)
        self.untried_moves.remove(move)
        if self.ai_color == COLOR_WHITE:
            move = Board_As_Black(
                -move.chessboard,
                move.chessboard_size,
                move.parent,
                move.last_move,
                move.depth
            )
        child_node = MCTSNode(move, parent=self, last_move=move.last_move, ai_color=-self.ai_color)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        if result == BLACK_WIN:
            self.wins += 1
        elif result == 0:
            self.wins += 0.5

    def one_add(self, result):
        if result == BLACK_WIN:
            self.wins += 1
        elif result == WHITE_WIN:     
            self.wins -= 0.5
    
    def tri_add(self, result):
        if result == BLACK_WIN:
            self.wins += 15
        elif result == WHITE_WIN:
            self.wins -= 10

    def kill(self):
        self.wins += -1000

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
        new_board = get_new_board(self.chessboard.copy(), target)
        if new_board is not None:
            new_board = Board_As_Black(
                new_board,
                self.chessboard_size,
                self,
                target,
                self.depth + 1
            )
        else:
            new_board = None
        return new_board

    def get_empty_positions(self):
        return list(zip(*np.where(self.chessboard == COLOR_NONE)))

    def get_whether_out_of_range(self, x_position, y_position):
        return x_position < 0 or x_position >= self.chessboard_size or y_position < 0 or y_position >= self.chessboard_size

def Determine_If_Need_Search(board_as_black):
    for move in board_as_black.get_child_board():
        if move.last_move in WIN_POSITION:
            return True, move.last_move
    return False, None
class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.mcts_time_limit = 4.2  # MCTS搜索时间限制(秒)

    def go(self, chessboard):
        self.candidate_list.clear()
        chessboard_as_black = self.adjust_board_as_black(chessboard)
        board_as_black = Board_As_Black(chessboard_as_black, self.chessboard_size)

        for available_child in board_as_black.get_child_board():
            self.candidate_list.append(available_child.last_move)
        if not self.candidate_list:
            return self.candidate_list

        empty_count = len(board_as_black.get_empty_positions())
        whether_need_to_search, last_move = Determine_If_Need_Search(board_as_black)
        if whether_need_to_search:
            best_move = last_move
        else:
            if empty_count >= 7:  # 早期使用mcts
                best_move = self.mcts_search(board_as_black)
            else:  # 后期使用更深的minimax
                _, best_move = self.minimax(board_as_black, 8, MODE_FLAT)
        if best_move:
            self.candidate_list.append(best_move)
        return self.candidate_list

    def mcts_search(self, root_board):
        """
        蒙特卡洛树搜索主函数
        参数:
            root_board: 当前棋盘状态(Board_As_Black对象)
        返回:
            最佳移动位置 (row, col)
        """
        # 创建根节点，记录开始时间
        root_node = MCTSNode(root_board, ai_color=COLOR_BLACK)
        start_time = time.time()
        cont = 0

        # 迭代搜索直到时间耗尽
        while time.time() - start_time < self.mcts_time_limit:
            cont += 1
            node = root_node
            mid_break = False

            while node.children and not node.get_untried_moves():
                node = node.select_child()
                if node.board.last_move in WIN_POSITION and node is not root_node:
                    if node.ai_color == COLOR_BLACK and len(root_node.board.get_empty_positions()) - len(node.board.get_empty_positions()) < 4:
                        result = BLACK_WIN
                        history_node = node
                        while node is not None:
                            node.one_add(result)
                            node = node.parent
                        node = history_node
                        if node.parent is root_node:
                            node.tri_add(result)
                            node.visits += 1
                            mid_break = True
                        node = history_node
                # elif node.board.last_move in LOSE_POSITION and len(root_node.board.get_empty_positions()) - len(node.board.get_empty_positions()) < 4 and node is not root_node:
                #     if node.ai_color == COLOR_BLACK:
                #         result = WHITE_WIN
                #         history_node = node
                #         while node is not None:
                #             node.one_add(result)
                #             node = node.parent
                #         node = history_node
                #         if node.parent is root_node:
                #             node.tri_add(result)
                #             node.visits += 1
                #             mid_break = True
                #         node = history_node
                elif node.board.last_move in LOSE_POSITION and len(root_node.board.get_empty_positions()) - len(node.board.get_empty_positions()) == 1 and node is not root_node:
                    node.kill()
                    result = WHITE_WIN
                    node.visits += 1
                    mid_break = True
                    break
            if mid_break:
                continue

            result = None
            if not node.get_untried_moves():
                if len(node.board.get_empty_positions()) - len(root_board.get_empty_positions()) < 4:
                    if node.board.last_move in WIN_POSITION:
                        if node.ai_color == COLOR_BLACK:
                            result = BLACK_WIN
                    elif node.board.last_move in LOSE_POSITION:
                        if node.ai_color == COLOR_BLACK:
                            result = WHITE_WIN
                    else: result = self.simulate(node.board, node.ai_color)
                else:
                    result = self.simulate(node.board, node.ai_color)
            else:
                node = node.expand()  # 这会自动切换玩家颜色
                result = self.simulate(node.board, node.ai_color)
            # 4. 回溯阶段 - 更新路径上所有节点的统计信息
            while node is not None:
                node.update(result)
                node = node.parent

        # 选择访问次数最多的子节点作为最佳移动
        if not root_node.children:
            return None

        # 返回访问次数最多的移动
        best_child = max(root_node.children, key=lambda c: c.visits)
        print(f"迭代次数: {cont}, 最佳移动: {best_child.last_move}, 访问次数: {best_child.visits}")
        return best_child.last_move

    def simulate(self, board, current_color):
        """
        随机模拟游戏直到终局
        参数:
            board: 当前棋盘状态
            current_color: 当前应该落子的玩家颜色
        返回:
        """
        current_board = board
        sim_color = current_color  # 当前模拟玩家颜色
        flipped_number = 0
        while True:
            empty_positions = current_board.get_empty_positions()
            # 如果没有空位，游戏结束
            if not empty_positions:
                break
            # 找到所有合法的落子位置
            child_boards = current_board.get_child_board()
            # 如双方没有合法的落子位置，游戏结束
            if not child_boards:
                flipped_board = Board_As_Black(
                    self.flip_the_board(current_board.chessboard),
                    self.chessboard_size,
                    current_board.parent,
                    current_board.last_move
                )
                if len(flipped_board.get_child_board()) == 0:
                    break
            # 如果当前玩家没有合法的落子位置，交换玩家并继续
            if not child_boards:
                fliped_board = Board_As_Black(
                    self.flip_the_board(current_board.chessboard),
                    self.chessboard_size,
                    current_board.parent,
                    current_board.last_move
                )
                flipped_number += 1
                current_board = fliped_board
                continue
            # 随机选择一个合法的落子位置
            random.shuffle(child_boards)
            current_board = random.choice(child_boards)
            fliped_board = Board_As_Black(
                self.flip_the_board(current_board.chessboard),
                self.chessboard_size,
                current_board.parent,
                current_board.last_move
            )
            flipped_number += 1
        if flipped_number % 2 == 1:
            current_board = Board_As_Black(
                self.flip_the_board(current_board.chessboard),
                self.chessboard_size,
                current_board.parent,
                current_board.last_move
            )
        totalResult = np.sum(current_board.chessboard)
        if sim_color == COLOR_BLACK:
            if totalResult > 0:
                return BLACK_WIN
            elif totalResult < 0:
                return WHITE_WIN
            else:
                return 0
        else:
            if totalResult > 0:
                return WHITE_WIN
            elif totalResult < 0:
                return BLACK_WIN
            else:
                return 0

    def minimax(self, board_as_black, max_depth, mode):
        board_min = Board_As_Black(self.flip_the_board(board_as_black.chessboard), self.chessboard_size)

        def max_value(board_min, alpha, beta, max_depth):
            board_max = Board_As_Black(
                self.flip_the_board(board_min.chessboard),
                self.chessboard_size,
                board_min.parent,
                board_min.last_move,
                board_min.depth
            )
            if self.terminal_test(board_max):
                return self.utility_for_max(board_max, mode), None
            if board_max.depth >= max_depth:
                return self.utility_for_max(board_max, mode), None

            v, best_move = -np.inf, None
            if len(board_max.get_child_board()) != 0:
                child_boards = board_max.get_child_board()
                random.shuffle(child_boards)
                for child in child_boards:
                    min_choice, _ = min_value(child, alpha, beta, max_depth)
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
            board_min = Board_As_Black(
                self.flip_the_board(board_max.chessboard),
                self.chessboard_size,
                board_max.parent,
                board_max.last_move,
                board_max.depth
            )
            if self.terminal_test(board_min):
                return self.utility_for_min(board_min, mode), None
            if board_min.depth >= max_depth:
                return self.utility_for_min(board_min, mode), None

            v, best_move = +np.inf, None
            if len(board_min.get_child_board()) != 0:
                child_boards = board_min.get_child_board()
                random.shuffle(child_boards)
                for child in child_boards:
                    max_choice, _ = max_value(child, alpha, beta, max_depth)
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

    def count_empty_positions(self, board_as_black):
        return len(board_as_black.get_empty_positions())

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
                fliped_board = Board_As_Black(
                    self.flip_the_board(board_as_black.chessboard),
                    self.chessboard_size,
                    board_as_black.parent,
                    board_as_black.last_move
                )
                if len(fliped_board.get_child_board()) == 0:
                    return True
        return False

    def utility_for_max(self, board_as_black, mode):
        if mode == MODE_FLAT:
            return np.sum(board_as_black.chessboard)
        elif mode == MODE_WEIGHTED:
            occupied = self.chessboard_size**2 - self.count_empty_positions(board_as_black)
            return np.sum(board_as_black.chessboard * WEIGHT_ONE)/max(1, occupied) + np.exp(0.6*np.sum(board_as_black.chessboard))
        elif mode == MODE_WEIGHTED_FREE:
            occupied = self.chessboard_size**2 - self.count_empty_positions(board_as_black)
            return np.sum(board_as_black.chessboard * WEIGHT_ONE)/max(1, occupied) - np.exp(0.6*len(board_as_black.get_child_board())) + np.exp(0.6*np.sum(board_as_black.chessboard))

    def utility_for_min(self, board_as_black, mode):
        if mode == MODE_FLAT:
            return -np.sum(board_as_black.chessboard)
        elif mode == MODE_WEIGHTED:
            occupied = self.chessboard_size**2 - self.count_empty_positions(board_as_black)
            return -np.sum(board_as_black.chessboard * WEIGHT_ONE)/max(1, occupied) - np.exp(0.4*np.sum(board_as_black.chessboard))
        elif mode == MODE_WEIGHTED_FREE:
            occupied = self.chessboard_size**2 - self.count_empty_positions(board_as_black)
            return -np.sum(board_as_black.chessboard * WEIGHT_ONE)/max(1, occupied) - np.exp(0.6*len(board_as_black.get_child_board())) - np.exp(0.4*np.sum(board_as_black.chessboard))
@njit
def get_new_board(board, target):
    x, y = target
    board[x][y] = COLOR_BLACK

    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    flipped = False

    for dx, dy in directions:
        tx, ty = x + dx, y + dy
        if not (0 <= tx < len(board) and 0 <= ty < len(board)):
            continue
        if board[tx][ty] != COLOR_WHITE:
            continue

        path = []
        while True:
            tx += dx
            ty += dy
            if not (0 <= tx < len(board) and 0 <= ty < len(board)):
                break
            if board[tx][ty] == COLOR_BLACK:
                flipped = True
                for px, py in path:
                    board[px][py] = COLOR_BLACK
                break
            elif board[tx][ty] == COLOR_WHITE:
                path.append((tx, ty))
            else:
                break

    return board if flipped else None
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
