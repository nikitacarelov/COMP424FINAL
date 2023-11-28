# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def computeAllMoves(self, player_pos, other_pos, max_step, chess_board):
        visited = set()
        allMoves = []
        num_squares_available = 0

        def dfs(x, y, steps_taken):
            nonlocal num_squares_available
            if steps_taken == max_step:
                return
            visited.add((x, y))
            num_squares_available += 1

            # which direction can we add walls for each square?
            for dir_wall in range(4):
                if not chess_board[x, y, dir_wall]:
                    board = deepcopy(chess_board)
                    # board = chess_board
                    board[x][y][dir_wall] = True
                    allMoves.append((((x, y), dir_wall), board))
                    # board[x][y][dir_wall] = False
                    

            # Moves (Up, Right, Down, Left)
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

            for (r, c, direction) in [(x - 1, y, 0), (x + 1, y, 2), (x, y - 1, 3), (x, y + 1, 1)]:
                if (r, c) not in visited and not chess_board[x, y, direction] and not other_pos == (r, c) and 0 <= r < len(chess_board) and 0 <= c < len(chess_board[0]):
                    dfs(r, c, steps_taken + 1)

        x, y = player_pos
        dfs(x, y, 0)

        return (allMoves, num_squares_available)

    def minimax(self, player_pos, dir, other_pos, max_step, chess_board, depth, max_player,
                alpha, beta):

        # minimax score heuristic: number squares I have to move - number squares adversay has to move in
        # depth limited minimax w AB pruning
        if depth == 1:
            nums_squares_I_have_to_move = self.computeAllMoves(
                player_pos, other_pos, max_step, chess_board)[1]

            nums_squares_adv_have_to_move = self.computeAllMoves(
                other_pos, player_pos, max_step, chess_board)[1]
            pos = player_pos if max_player else other_pos
            return (nums_squares_I_have_to_move - nums_squares_adv_have_to_move, (pos, dir))

        if max_player:
            best = (-1000, ((0, 0), 0))

            # compute all moves
            moves_list = self.computeAllMoves(
                player_pos, other_pos, max_step, chess_board)[0]
            for (((max_player_pos, dir), board)) in moves_list:

                # call minimax on all children of node
                minimax_res = self.minimax(max_player_pos, dir, other_pos, max_step,
                                           board, depth + 1, False, alpha, beta)
                val = minimax_res[0]
                best = best if max(
                    best[0], val) == best[0] else minimax_res

                alpha = alpha if max(
                    alpha[0], best[0]) == alpha[0] else minimax_res
                # alpha = max(alpha, best)

                # Alpha Beta Pruning
                if beta[0] <= alpha[0]:
                    break
            return best

        else:
            best = (1000, ((0, 0), 0))

            # Recur for left and
            # right children
            moves_list = self.computeAllMoves(
                other_pos, player_pos, max_step, chess_board)[0]

            for (((min_player_pos, dir), board)) in moves_list:
                minimax_res = self.minimax(player_pos, dir, min_player_pos, max_step,
                                           board, depth + 1, True, alpha, beta)

                val = minimax_res[0]

                best = best if min(
                    best[0], val) == best[0] else minimax_res

                beta = beta if min(
                    alpha[0], best[0]) == alpha[0] else minimax_res
                # beta = min(beta, best)

                # Alpha Beta Pruning
                if beta[0] <= alpha[0]:
                    break

        return best

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.

        start_time = time.time()
        # first compute all moves for my_pos since its my turn

        minimax_res = self.minimax(my_pos, 0, adv_pos, max_step, chess_board, 0, True,
                                   (-1000, ((0, 0), 0)), (1000, ((0, 0), 0)))
        print("result of minimax: ", minimax_res)

        time_taken = time.time() - start_time

        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return minimax_res[1]
        # return my_pos, self.dir_map["u"]
