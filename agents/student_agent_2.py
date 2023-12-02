# Student agent: Add your own agent here
import random

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import math


@register_agent("student_agent_2")
class StudentAgent2(Agent):
    def __init__(self):
        super(StudentAgent2, self).__init__()
        self.name = "StudentAgent2"
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

        # Verifies that the requested move is feasible
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

        # TODO: Check that check_endgame works as expected
        def check_endgame(board, my_pose, adv_pose):
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
                        if board[r, c, dir + 1]:
                            continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)

            for r in range(board_size):
                for c in range(board_size):
                    find((r, c))
            p0_r = find(tuple(my_pose))
            p1_r = find(tuple(adv_pose))
            p0_score = list(father.values()).count(p0_r)
            p1_score = list(father.values()).count(p1_r)
            if p0_r == p1_r:
                return False, p0_score, p1_score
            # player_win = None
            # win_blocks = -1
            # if p0_score > p1_score:
            #     player_win = 0
            #     win_blocks = p0_score
            # elif p0_score < p1_score:
            #     player_win = 1
            #     win_blocks = p1_score
            # else:
            #     player_win = -1  # Tie
            # if player_win >= 0:
            #   print(f"Game ends! Player {self.player_names[player_win]} wins having control over {win_blocks} blocks!"
            #    )
            # else
            #    print("Game ends! It is a Tie!")

            return True, p0_score, p1_score

        # Returns a copy of the chess_board with the new location of the wall
        def update_board(board, new_x, new_y, d):
            new_board = deepcopy(board)
            # Assuming the board is represented as a numpy array with True indicating wall positions
            new_board[new_x][new_y][d] = True
            return new_board

        # heuristic score evaluationm for player calling the function
        def heuristic_evaluation(board, my_pose, adv_pose, d):
            def rom_heuristic():
                my_rom = get_rom(my_pose)
                adv_rom = get_rom(adv_pos)
                my_score = 0
                adv_score = 0
                for square in my_rom:
                    for d in range(4):
                        if check_valid_move(my_pose, np.array(square), d, adv_pose):
                            my_score += 1

                for square in adv_rom:
                    for d in range(4):
                        if check_valid_move(adv_pose, np.array(square), d, my_pose):
                            adv_score += 1

                score = my_score - adv_score
                return score

            def distance_heuristic():
                dist = np.sum(math.dist(my_pose, adv_pose))
                dist_score = abs(30 / dist)
                return dist_score

            def wall_heuristic():
                wall_score = 0
                long_score = 7
                corner_score = 4
                death_score = -200

                # YOU cornered: BAD
                if np.sum(board[my_pose[0], my_pose[1]]) == 3:
                    wall_score = death_score

                # Horizontal Line:
                if d == 0 or d == 2:  # walls in line with up or down wall
                    try:
                        if board[my_pose[0], my_pose[1] + 1, d] == 1:
                            wall_score += long_score
                    except:
                        pass
                    try:
                        if board[my_pose[0], my_pose[1] - 1, d] == 1:
                            wall_score += long_score
                    except:
                        pass

                # Up Corner:
                if d == 0:  # walls left or right upwards of you
                    try:
                        if board[my_pose[0] - 1, my_pose[1], 1] == 1 or board[my_pose[0] - 1, my_pose[1], 3] == 1:
                            wall_score += corner_score
                    except:
                        pass

                # Down Corner:
                if d == 2:  # walls left or right downwards of you
                    try:
                        if board[my_pose[0] + 1, my_pose[1], 1] == 1 or board[my_pose[0] + 1, my_pose[1], 3] == 1:
                            wall_score += corner_score
                    except:
                        pass

                # Vertical Line:
                if d == 1 or d == 3:  # right or left walls
                    try:
                        if board[my_pose[0] + 1, my_pose[1], d] == 1:
                            wall_score += long_score
                    except:
                        pass
                    try:
                        if board[my_pose[0] - 1, my_pose[1], d] == 1:
                            wall_score += long_score
                    except:
                        pass

                # Right Corner:
                if d == 1:  # walls down or up rightwards of you
                    try:
                        if board[my_pose[0], my_pose[1] + 1, 0] == 1 or board[my_pose[0], my_pose[1] + 1, 2] == 1:
                            wall_score += corner_score
                    except:
                        pass

                # Left Corner:
                if d == 3:  # walls left or right downwards of you
                    try:
                        if board[my_pose[0], my_pose[1] - 1, 0] == 1 or board[my_pose[0], my_pose[1] - 1, 2] == 1:
                            wall_score += corner_score
                    except:
                        pass
                return wall_score

            def endgame_heuristic():
                end_return = check_endgame(board, my_pose, adv_pose)
                if end_return[0] is True:
                    end_score = end_return[1] - end_return[2]
                    print("end score: ", end_score)
                    return end_score
                else:
                    return 0

            score = ((2 * rom_heuristic()) + (20 * distance_heuristic()) + (2 * wall_heuristic()) +
                     (100 * endgame_heuristic()))
            return score

        # Get a list of squares accessible ignoring any walls
        def get_rom(position):
            rom = []
            for x in range(-max_step, max_step + 1):
                for y in range(-max_step, max_step + 1):
                    # if the step is within range of the agent
                    if abs(x) + abs(y) <= max_step:
                        # if the movement is within the map
                        if (
                                0 <= position[0] + x < board_size
                                and 0 <= position[1] + y < board_size
                        ):
                            rom.append([position[0] + x, position[1] + y])
                            # Convert the list of pairs to a NumPy array
            # rom = np.array(rom).T if rom else np.array([[], []])
            return rom

        def get_game_state(board):
            mid_game_ratio = 0.2
            end_game_ratio = 0.4

            num_walls = np.sum(board)
            max_walls = board_size * board_size * 4 - (board_size * 2)
            ratio = num_walls / max_walls
            if 0 < ratio <= mid_game_ratio:
                return 1, ratio
            if mid_game_ratio < ratio <= end_game_ratio:
                return 2, ratio
            if end_game_ratio < ratio:
                return 3, ratio

        def change_rom(rom, my_pose, adv_pose, maximizer):
            new_rom = rom
            if game_state[0] == 1:
                for square in new_rom:
                    if math.dist(square, adv_pose) > math.dist(my_pose, adv_pose):
                        # if random.random() < 0.5:
                        new_rom.remove(square)
                np.random.shuffle(new_rom)
                if not maximizer:
                    new_rom = new_rom[0:(len(new_rom) // 3)]

            elif game_state[0] == 2:
                np.random.shuffle(new_rom)
                new_rom = new_rom[0:(len(new_rom) // 2)]

            return new_rom

        def change_dirs(dirs, square, adv_pose):
            game_state, ratio = get_game_state(chess_board)
            new_dirs = dirs
            if game_state == 1:
                if square[0] - np.array(adv_pose[0], adv_pose[1]) < 1:
                    if random.random() < 0.3:
                        new_dirs.remove(2)
                else:
                    if random.random() < 0.3:
                        new_dirs.remove(0)

                if square[1] - np.array(adv_pose[0], adv_pose[1]) < 1:
                    if random.random() < 0.3:
                        new_dirs.remove(3)
                else:
                    if random.random() < 0.3:
                        new_dirs.remove(1)
            return new_dirs

        # TODO: fix minimax algorithm
        # Main recursive minimax function:
        def depth_limited_minimax(board, my_pose, adv_pose, depth, alpha, beta, max_time, wall_dir,
                                  maximizing_player=True):

            if depth == 0 or (time.time() - start_time) > max_time:
                if maximizing_player:
                    return (my_pose, wall_dir), heuristic_evaluation(board, my_pose, adv_pose, None)
                else:
                    return (my_pose, wall_dir), heuristic_evaluation(board, adv_pose, my_pose, None)
            # end_return = check_endgame(board, my_pose, adv_pose)
            #
            # if end_return[0]:
            #     score = heuristic_evaluation(board, my_pose, adv_pose, None)
            #     return (my_pose, wall_dir), score

            # Maximizer logic
            if maximizing_player:
                max_eval = float(-10000)  # GOOD
                best_move = ((13, 13), 3)
                rom = get_rom(my_pose)
                # rom = change_rom(rom, my_pose, adv_pose, True)
                # Iterating over every square and every wall in rom
                for square in rom:
                    dirs = [0, 1, 2, 3]
                    np.random.shuffle(dirs)
                    # new_dirs = change_dirs(dirs, square, adv_pose)
                    for d in dirs:
                        if time.time() - start_time > max_time:
                            return best_move, max_eval
                        else:
                            # If move valid, proceed
                            if check_valid_move(my_pose, np.array(square), d, adv_pose):
                                new_my_pos = square
                                # Update the board with the move
                                new_chess_board = update_board(board, square[0], square[1], d)
                                print("maximizer")  # Debug
                                score = heuristic_evaluation(new_chess_board, new_my_pos, adv_pose, d)
                                print("Move: ", (new_my_pos, d), "score: ", score)
                                if score > 0:  # or random.random() > 0.2:
                                    # print("Entering Maximizer Move with score: [", score, "] and depth: [", depth, "]")
                                    # Perform recursive depth-limited Minimax
                                    _, maximizer_eval = depth_limited_minimax(new_chess_board, new_my_pos, adv_pose,
                                                                              depth - 1, alpha, beta, max_time, d,
                                                                              False)
                                else:
                                    maximizer_eval = score

                                if maximizer_eval > max_eval:
                                    max_eval = maximizer_eval
                                    best_move = ((new_my_pos[0], new_my_pos[1]), d)
                                alpha = max(alpha, maximizer_eval)
                                if beta <= alpha:
                                    break  # Beta cut-off

                return best_move, max_eval

            else:  # Minimizing player (adversary)
                min_eval = float(10000)
                best_move = ((13, 13), 3)
                rom = get_rom(adv_pose)
                # rom = change_rom(rom, adv_pose, my_pose, False)
                # Iterating over every square and every wall in ROM
                for square in rom:
                    dirs = [0, 1, 2, 3]
                    np.random.shuffle(dirs)
                    for d in dirs[0:4]:
                        if time.time() - start_time > max_time:
                            return best_move, min_eval

                        else:
                            # If move valid, proceed
                            if check_valid_move(adv_pose, np.array(square), d, my_pose):

                                new_adv_pos = square

                                # Update the board with the move
                                new_chess_board = update_board(board, square[0], square[1], d)
                                print("Minimizer")  # debug
                                score = heuristic_evaluation(new_chess_board, new_adv_pos, my_pose, d)
                                if score < 0:
                                    # print("Entering Minimizer Move with score: [", score, "] and depth: [", depth, "]")

                                    # Perform recursive depth-limited Minimax
                                    _, minimizer_eval = depth_limited_minimax(new_chess_board, my_pose, new_adv_pos,
                                                                              depth - 1, alpha,
                                                                              beta, max_time, d, True)
                                else:  # For when you can't find a positive eval
                                    minimizer_eval = score

                                if minimizer_eval < min_eval:
                                    min_eval = minimizer_eval
                                    best_move = ((new_adv_pos[0], new_adv_pos[1]), d)

                                beta = min(beta, minimizer_eval)
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

        def check_max_time(t, max_time_taken):
            if t - time.time() > max_time_taken:
                max_time_taken = t - time.time()
                return True, max_time_taken
            else:
                return False



        start_time = time.time()
        game_state = get_game_state(chess_board)

        depth_limit = 1
        best_move = None
        max_time = 2
        best_eval = 0

        while time.time() - start_time < max_time:  # Time limit of 5 seconds
            alpha = float('-inf')
            beta = float('inf')

            # temp_best_move = None

            # Perform depth-limited Minimax with iterative deepening
            new_best_move, eval = depth_limited_minimax(chess_board, my_pos, adv_pos, depth_limit, alpha, beta,
                                                        max_time, True)

            if eval > best_eval:
                temp_best_move = new_best_move
                best_move = temp_best_move
                print("best move: ", best_move, " with score: ", eval)
            # Update the depth limit for the next iteration
            depth_limit += 1

        if best_move is None:
            return (13, 13), 1

        print("this took ", time.time() - start_time, " seconds")
        return best_move

        # dummy return
        # return my_pos, self.dir_map["u"]
