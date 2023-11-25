# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import math


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
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def step(self, chess_board, my_pos, adv_pos, max_step):
        # Assuming player is p0

        board_size = len(chess_board[0])

        def check_valid_move(start_pos, end_pos, barrier_dir, adv_pos):
            """
            Check if the step the agent takes is valid (reachable and within max steps).

            Parameters
            ----------
            start_pos : tuple
                The start position of the agent.
            end_pos : np.ndarray
                The end position of the agent.
            barrier_dir : int
                The direction of the barrier.
            """
            # Endpoint already has barrier or is border

            r, c = end_pos
            if chess_board[r, c, barrier_dir]:
                return False
            if np.array_equal(start_pos, end_pos):
                return True

            # Get position of the adversary
            # adv_pos = self.p0_pos if self.turn else self.p1_pos

            # BFS
            state_queue = [(start_pos, 0)]
            visited = {tuple(start_pos)}
            is_reached = False
            while state_queue and not is_reached:
                cur_pos, cur_step = state_queue.pop(0)

                r, c = cur_pos
                if cur_step == max_step:
                    break
                for dir, move in enumerate(self.moves):
                    if chess_board[r, c, dir]:
                        continue

                    next_pos = np.array(cur_pos) + move
                    if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                        continue
                    if np.array_equal(next_pos, end_pos):
                        is_reached = True
                        break

                    visited.add(tuple(next_pos))
                    state_queue.append((next_pos, cur_step + 1))

            return is_reached

        # https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-1-introduction/

        def check_endgame(chess_board):
            """
            Check if the game ends and compute the current score of the agents.

            Returns
            -------
            is_endgame : bool
                Whether the game ends.
            player_1_score : int
                The score of player 1.
            player_2_score : int
                The score of player 2.
            """
            # Union-Find
            father = dict()
            for r in range(board_size):
                for c in range(board_size):
                    father[(r, c)] = (r, c)

            def find(pos):
                if father[pos] != pos:
                    father[pos] = find(father[pos])
                return father[pos]

            def union(pos1, pos2):
                father[pos1] = pos2

            for r in range(board_size):
                for c in range(board_size):
                    for dir, move in enumerate(
                            self.moves[1:3]
                    ):  # Only check down and right
                        if chess_board[r, c, dir + 1]:
                            continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)

            for r in range(board_size):
                for c in range(board_size):
                    find((r, c))
            p0_r = find(tuple(my_pos))
            p1_r = find(tuple(adv_pos))
            p0_score = list(father.values()).count(p0_r)
            p1_score = list(father.values()).count(p1_r)
            if p0_r == p1_r:
                return False, p0_score, p1_score
            player_win = None
            win_blocks = -1
            if p0_score > p1_score:
                player_win = 0
                win_blocks = p0_score
            elif p0_score < p1_score:
                player_win = 1
                win_blocks = p1_score
            else:
                player_win = -1  # Tie
            # if player_win >= 0:
            #   print(f"Game ends! Player {self.player_names[player_win]} wins having control over {win_blocks} blocks!"
            #    )
            # else:
            #    print("Game ends! It is a Tie!")
            return True, p0_score, p1_score

        def update_board(chess_board, new_x, new_y, d):
            new_chess_board = deepcopy(chess_board)
            # Assuming the board is represented as a numpy array with True indicating occupied positions
            new_chess_board[new_x][new_y][d] = 1
            return new_chess_board

        def heuristic_evaluation(chess_board, my_pos, adv_pos):
            # Replace this with your own heuristic function that evaluates the score for a given node
            # This function should return a heuristic score for the node at the given index

            # number of moves player
            my_valid_moves = 0
            for x in range(board_size):
                for y in range(board_size):
                    for d in range(4):
                        # If move valid, proceed
                        if check_valid_move(my_pos, np.array((x, y)), d, adv_pos):
                            my_valid_moves += 1

            # number of moves adversary
            adv_valid_moves = 0
            for x in range(board_size):
                for y in range(board_size):
                    for d in range(4):
                        # If move valid, increment moves
                        if check_valid_move(adv_pos, np.array((x, y)), d, adv_pos):
                            adv_valid_moves += 1

            score = my_valid_moves - adv_valid_moves
            return score

        def depth_limited_minimax(chess_board, my_pos, adv_pos, depth, alpha, beta, maximizing_player=True):
            if depth == 0:
                return None, heuristic_evaluation(chess_board, my_pos, adv_pos)

            end_return = check_endgame(chess_board)

            if end_return[0]:
                score = end_return[1] - end_return[2]
                return score
            # Maximizer logic
            if maximizing_player:
                max_eval = float('-inf')
                best_move = None

                # Iterating over every square and every wall on board
                for x in range(board_size):
                    for y in range(board_size):
                        for d in range(4):
                            # If move valid, proceed
                            if check_valid_move(my_pos, np.array((x, y)), d, adv_pos):
                                new_my_pos = (x, y)
                                print("checking valid move: ", x, ", ", y, ", ", d, "at depth", depth)
                                # copying the board and updating with the move
                                new_chess_board = update_board(chess_board, x, y, d)  # Update board with new wall

                                _, eval = depth_limited_minimax(new_chess_board, new_my_pos, adv_pos, depth - 1, alpha,
                                                                beta, False)

                                if eval > max_eval:
                                    max_eval = eval
                                    best_move = (new_my_pos, d)

                                alpha = max(alpha, eval)
                                if beta <= alpha:
                                    break  # Beta cut-off

                return best_move, max_eval

            else:  # Minimizing player (adversary)
                min_eval = float('inf')
                best_move = None

                # Iterating over every square and every wall on board
                for x in range(board_size):
                    for y in range(board_size):
                        for d in range(4):
                            # If move valid, proceed
                            if check_valid_move(adv_pos, np.array((x, y)), d, adv_pos):
                                new_adv_pos = (x, y)

                                # copying the board and updating with the move
                                new_chess_board = update_board(chess_board, x, y, d)  # Update board with new wall

                                _, eval = depth_limited_minimax(new_chess_board, my_pos, new_adv_pos, depth - 1, alpha,
                                                                beta, True)

                                if eval < min_eval:
                                    min_eval = eval
                                    best_move = (my_pos, d)  # Assuming adversary doesn't move

                                beta = min(beta, eval)
                                if beta <= alpha:
                                    break  # Alpha cut-off

                return best_move, min_eval

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
        depth_limit = 1
        best_move = None
        while time.time() - start_time < 5:  # Time limit of 2 seconds
            # Perform iterative deepening with depth-limited Minimax
            alpha = float('-inf')
            beta = float('inf')
            temp_best_move = None

            # Iterative deepening Minimax search
            for depth in range(1, max_step + 1):
                # Call the depth-limited Minimax function with depth, alpha, beta, etc.
                new_best_move = depth_limited_minimax(chess_board, my_pos, adv_pos, depth, alpha, beta)

                if new_best_move:
                    temp_best_move = new_best_move
                    best_move = temp_best_move
                else:
                    break

            # Update depth_limit based on time and other criteria if needed
            depth_limit += 1
        # time_taken = start_time - time.time()
        # print("My AI's turn took ", time_taken, "seconds.")

        return best_move[0]

        # dummy return
        # return my_pos, self.dir_map["u"]
