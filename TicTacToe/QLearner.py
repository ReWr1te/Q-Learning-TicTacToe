from Board import Board
import numpy as np


class QLearner:
    """  Your task is to implement `move()` and `learn()` methods, and 
         determine the number of games `GAME_NUM` needed to train the qlearner
         You can add whatever helper methods within this class.
    """

    # ======================================================================
    # ** set the number of games you want to train for your qlearner here **
    # ======================================================================
    GAME_NUM = 100000

    def __init__(self):
        """ Do whatever you like here. e.g. initialize learning rate
        """
        # =========================================================
        #  
        # 
        # ** Your code here **
        # q values attributes
        self.Q = {}  # q_values
        self.actions = []  # actions of last game
        self.alpha = 0.3  # learning rate
        self.gamma = 0.9  # discount rate
        self.init_q = 0.0  # initial q value
        self.grids = 9  # number of grids
        # usage of q_values control factors
        self.epsilon = 1.0  
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay = 0.01  # decay of the weight of the number of games
        self.first = False  # moved first
        self.games = 0  # game numbers
        #
        # 
        # =========================================================     

    def move(self, board):
        """ given the board, make the 'best' move 
            currently, qlearner behaves just like a random player  
            see `play()` method in TicTacToe.py 
        Parameters: board 
        """
        if board.game_over():
            return

        # =========================================================
        # ** Replace Your code here  **
        # find all legal moves
        cands = [i for i in range(9) if board.is_valid_move(i//3, i%3)]
        if cands == 9:  # first move judgement
            self.first = True
        state = tuple(board.state.flatten())
        if len(cands) == 9:  # if move first, move to corners first
            move = [0, 2, 6, 8][np.random.randint(4)]
        elif len(cands) == 8:  # if move second, do not move to edges
            move = 4 if 4 in cands else [0, 2, 6, 8][np.random.randint(4)]
        else:  # later actions
            # give some space to explore at the early stage
            if np.random.uniform(0, 1) > self.epsilon:  # exploit
            # if True:
                if state not in self.Q:  # no related q value, explore
                    self.Q[state] = [self.init_q] * self.grids
                    move = cands[np.random.randint(len(cands))]
                else:  # exploit
                    move = cands[np.argmax([self.Q[state][m] for m in cands])]
            else:  # explore
                move = cands[np.random.randint(len(cands))]
        self.actions.append((state, move))  # record for learning
        move = (move//3, move%3)
        # =========================================================

        return board.move(move[0], move[1], self.side)   

    def update_Q(self, l_s, l_m, s, reward):
        """ Update q value according to last state, last move, current state
            and reward.
        Parameters:
            l_s: tuple, last state
            l_m: tuple, last move
            s: tuple, current state
            reward: float, reward value
        """
        if l_m >= 0:
            if l_s not in self.Q:  # first time to meet this state
                self.Q[l_s] = [self.init_q] * self.grids
                cur_q = 0.0
            else:
                cur_q = self.Q[l_s][l_m]  # get current q value
            # the core of q learning method
            next_moves_dic = [self.init_q] * self.grids
            max_q_next = np.max(self.Q.get(s, next_moves_dic))
            self.Q[l_s][l_m] = (1 - self.alpha) * cur_q
            self.Q[l_s][l_m] += self.alpha * (reward + self.gamma * max_q_next)

    def learn(self, board):
        """ when the game ends, this method will be called to learn from the previous game i.e. update QValues
            see `play()` method in TicTacToe.py 
        Parameters: board
        """
        # =========================================================
        #  
        # 
        # ** Your code here **
        winner = board._check_winner()
        # set reward according to current side and winner
        reward_dic = {0: 0.5, self.side: 1.0, 1 if 1 != self.side else 2: -1.0}
        # check whether Q Learner made the last move
        flag_last = self.actions[-1][0] == tuple(board.state.flatten())
        # update q values
        i = 0
        while i < len(self.actions) - (2 if flag_last else 1):
            l_s, l_m= self.actions[i]
            s = self.actions[i+1][0]
            self.update_Q(l_s, l_m, s, 0.0)
            i += 1
        # update the last move q value
        l_s, l_m = self.actions[-(2 if flag_last else 1)]
        self.update_Q(l_s, l_m, tuple(board.state.flatten()), reward_dic[winner])
        # set attributes
        self.epsilon = self.min_epsilon + ((self.max_epsilon - self.min_epsilon)*
            np.exp(-self.decay * (self.games + 1)))
        self.games += 1
        self.actions = []
        self.first = False
        # =========================================================

    # do not change this function
    def set_side(self, side):
        self.side = side
