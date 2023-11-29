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

            # which direction can we add walls for each square?
            for dir_wall in range(0, 4):
                if not chess_board[x, y, dir_wall]:
                    board = deepcopy(chess_board)
                    board[x][y][dir_wall] = True
                    allMoves.append((((x, y), dir_wall), board))
                    num_squares_available += 1

            for (r, c, dir) in [(x - 1, y, 0), (x + 1, y, 2), (x, y - 1, 3), (x, y + 1, 1)]:
                if (r, c) not in visited and not chess_board[x, y, dir] and not other_pos == (r, c) and 0 <= r < len(chess_board) and 0 <= c < len(chess_board[0]):
                    dfs(r, c, steps_taken + 1)

        x, y = player_pos
        dfs(x, y,0)

        return (allMoves, num_squares_available)

    def minimax(self, player_pos, other_pos, max_step, chess_board, depth, max_player,
                alpha, beta):

        # minimax score heuristic: number squares I have to move - number squares adversary has to move in
        # depth limited minimax w AB pruning
        if depth == 2:
            nums_squares_I_have_to_move = self.computeAllMoves(
                player_pos, other_pos, max_step, chess_board)[1]

            nums_squares_adv_have_to_move = self.computeAllMoves(
                other_pos, player_pos, max_step, chess_board)[1]

            return nums_squares_I_have_to_move - nums_squares_adv_have_to_move

        if max_player:
            best = -1000

            # compute all moves
            moves_list = self.computeAllMoves(
                player_pos, other_pos, max_step, chess_board)[0]
            for ((max_player_pos, direction), board) in moves_list:

                # call minimax on all children of node
                val = self.minimax(max_player_pos, other_pos, max_step,
                                           board, depth + 1, False, alpha, beta)

                best = max(best, val)
                alpha = max(alpha, best)

                # Alpha Beta Pruning
                if beta <= alpha:
                    break
            return best

        else:
            best = 1000

            moves_list = self.computeAllMoves(
                other_pos, player_pos, max_step, chess_board)[0]

            for ((min_player_pos, direction), board) in moves_list:
                val = self.minimax(player_pos, min_player_pos, max_step,
                                   board, depth + 1, True, alpha, beta)

                best = min(best, val)
                beta = min(alpha, best)

                # Alpha Beta Pruning
                if beta <= alpha:
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
        # first compute all moves for my_pos since its my turn

        start_time = time.time()
        depth = 0
        alpha = -1000
        beta = 1000
        best_moves = []

        moves_list = self.computeAllMoves(my_pos, adv_pos, max_step, chess_board)[0]

        for ((max_player_pos, direction), board) in moves_list:
            # call minimax on all children of node
            val = self.minimax(max_player_pos, adv_pos, max_step,
                               board, depth + 1, False, alpha, beta)
            if val > alpha:
                alpha = val
                best_moves.clear()
                best_moves.append((max_player_pos, direction))
            elif val == alpha:
                best_moves.append((max_player_pos, direction))





        time_taken = time.time() - start_time

        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return best_moves[np.random.randint(0, len(best_moves))] if len(best_moves) > 1 else best_moves[0]



