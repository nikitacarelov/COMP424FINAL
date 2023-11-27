# Student agent: Add your own agent here
from collections import defaultdict
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
        self.moves_tree = defaultdict(dict)
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def computeAllMoves(self, player_pos, other_pos, max_step, chess_board, max_depth_tree):
        visited = set()
        #switch between whos turn we're computing

        def dfs(x, y, dir, steps_taken):
            
            if max_depth_tree <= 0:
                return
        
            if steps_taken == max_step + 1:
                return
            visited.add((x, y))

            # which direction can we add walls for each square?
            for dir_wall in range(0, 4):
                if not chess_board[x, y, dir_wall]:
                    board = deepcopy(chess_board)
                    board[x][y][dir] = True
                    
                    tmp = player_pos
                    p1_pos = other_pos
                    p2_pos = tmp
                    self.moves_tree[player_pos].update({(x, y, dir_wall):{self.computeAllMoves(p1_pos, p2_pos, max_step, board, max_depth_tree -1)}})

            for (r, c, dir) in [(x - 1, y, 0), (x + 1, y, 2), (x, y - 1, 3), (x, y + 1, 1)]:
                if (r, c) not in visited and not chess_board[x, y, dir] and not other_pos == (r, c) and 0 <= r < len(chess_board) and 0 <= c < len(chess_board[0]):
                    dfs(r, c, dir, steps_taken + 1)
                   
        x, y = player_pos
        dfs(x, y, 0, 0)
        return 
        
    

    def decideMove(self, player_pos, other_pos, max_step, chess_board):
        # def IDS(self, max_depth, player_pos, other_pos):
        #     for _ in range(max_depth):
        #         self.computeAllMoves(player_pos, other_pos, max_step, chess_board)

        # self.IDS(defaultdict(dict), 3, player_pos, other_pos)
        self.computeAllMoves(player_pos, other_pos, max_step, chess_board, 1)

        print(self.moves_tree)

       # minimax score heuristic: number squares I have to move - number squares adversay has to move in

       # run (a,b) pruning

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
        self.decideMove(my_pos, adv_pos, max_step, chess_board)

        time_taken = time.time() - start_time

        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]
