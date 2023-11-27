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
        self.moves_tree = defaultdict(set)
 
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }


    def computeAllMoves(self, player_pos, other_pos, max_step, chess_board):
        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        visited = set()
        
        def dfs(x, y, dir, depth):
            
            if depth == max_step + 1:
                return False
            visited.add((x,y))
            self.moves_tree[player_pos].add((x,y))
                
            for (r, c, dir) in [(x - 1, y, 0), (x + 1, y, 2), (x, y - 1, 3), (x, y + 1, 1)]:
                if depth != 0 and (r,c) not in visited and not chess_board[x, y, dir] and not other_pos == (r,c) and 0 <= r < len(chess_board) and 0 <= c < len(chess_board[0]):
                    if dfs(r, c, dir, depth + 1):
                        return True
            return False
        
        x, y = player_pos
        dfs(x, y, 0, 0)
        print(visited)
        print(self.moves_tree)
        
            
    def decideMove(self, player_pos, other_pos, max_step, chess_board):
        #first compute all moves for my_pos since its my turn
        #then compute advsery mvoes
        #then my
        #then ad
        
        self.computeAllMoves(player_pos, other_pos, max_step, chess_board)
        #run IDS and find best move
        # print(self.moves_tree)
    
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
        #first compute all moves for my_pos since its my turn
        self.decideMove(my_pos, adv_pos, max_step, chess_board)
        
        
        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]