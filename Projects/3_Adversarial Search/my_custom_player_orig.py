import pickle
import random
from collections import defaultdict, Counter
import numpy as np
from sample_players import DataPlayer
import sys

class Node:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.actions = list(state.actions())
        self.untried_actions = list(state.actions())
        random.shuffle(self.untried_actions)
        self.parent = parent
        self.children = []
        self.n = 0
        self.q = 0

    # def __repr__(self):
    #     return "({:>2},{:>3})".format(self.n, self.q)

    @property
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self):
        return len(self.actions) == 0
        # return self.state.terminal_test()


class Stats:
    def __init__(self, action=None):
        self.parent = action
        self.actions = []
        self.n = 0
        self.q = 0

    def __repr__(self):
        return "({:>2},{:>3})".format(self.n, self.q)


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only *required* method. You can modify
    the interface for get_action by adding named parameters with default
    values, but the function MUST remain compatible with the default
    interface.

    **********************************************************************
    NOTES:
    - You should **ONLY** call methods defined on your agent class during
      search; do **NOT** add or call functions outside the player class.
      The isolation library wraps each method of this class to interrupt
      search when the time limit expires, but the wrapper only affects
      methods defined on this class.

    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    **********************************************************************
    """
    def __init__(self, player_id):
        super().__init__(player_id)
        # self.data = {} if self.data is None else {k: max(v, key=v.get) for k, v in self.data.items()}

        self.data = {} if self.data is None else self.data
        self.ply_count = None
        # self.route = []
        # self.is_terminal_route = False
        # self.terminal_routes = []
        # self.end_of_sim_stats = None
        self.context = {
            # 'total_plays': 0,
            # 'total_branches': 0,
            # 'mcts': defaultdict(Stats),
            # 'symmetry_actions': create_symmetry_actions()
        }

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """

        depth = 3
        # self.state = state
        self.ply_count = state.ply_count
        # print(state.terminal_test())

        # if state in self.data:
        #     self.queue.put(self.data[state])
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
            # return random.choice(state.actions())
        else:
            # self.context['total_plays'] += 1
            # self.context['total_branches'] += len(state.actions())
            # print(self.ply_count, self.context['total_branches'], self.context['total_branches'] / self.context['total_plays'])
            # RANDOM
            # self.queue.put(random.choice(state.actions()))
            # GREEDY
            self.queue.put(max(state.actions(), key=lambda x: self.greedy_score(state.result(x))))
            # MINIMAX
            # self.queue.put(self.minimax(state, depth))
            # self.queue.put(max(state.actions(), key=lambda x: self.minimax1(state.result(x), depth - 1, False)))
            # self.queue.put(max(state.actions(), key=lambda x: self.minimax2(state.result(x), depth - 1, False)))
            # ALPHA-BETA
            # self.queue.put(max(state.actions(), key=lambda x: self.alpha_beta(state.result(x), depth - 1, float("-inf"), float("inf"), False)))
            # self.queue.put(max(state.actions(), key=lambda x: self.alpha_beta1(state.result(x), float("-inf"), float("inf"), False)))
            # self.queue.put(max(state.actions(), key=lambda x: self.alpha_beta2(state.result(x), depth - 1, float("-inf"), float("inf"), False)))
            # MCTS-UCT
            # self.queue.put(self.uct_search(state))
            # self.queue.put(self.uct_search_using_dict())
            # return self.uct_search(state)
            # return self.uct_search_using_dict()

    # UCT USING LINKED LIST

    def uct_search(self, state):
        root_node = Node(state)
        for n in range(80):
            node = self.tree_policy(root_node)
            reward = self.default_policy(node.state)
            self.backup_negamax(node, reward)
            # print('{:>3}'.format(n), self.end_of_sim_stats, end=' ')
            # print([child for child in root_node.children], end=' ')
            # route = '_'.join(list(map(str, self.route)))
            # print(route)
            # if self.is_terminal_route:
            #     if route in self.terminal_routes:
            #         dummy = ''
            #     else:
            #         self.terminal_routes.append(route)
            # self.is_terminal_route = False
            # self.route = []
        return self.best_child(root_node, 0).action

    def tree_policy(self, node):
        while not node.is_terminal:
            if not node.is_fully_expanded:
                return self.expand(node)
            else:
                node = self.best_child(node)
        # self.route.append('x')
        # self.is_terminal_route = True
        return node

    def expand(self, node):
        action = node.untried_actions.pop()
        child = Node(node.state.result(action), action, node)
        # self.route.append(len(node.children))
        node.children.append(child)
        return child

    def best_child(self, node, c=1/np.sqrt(2)):
        if c == 0:
            best = max(node.children, key=lambda s: s.q / s.n)
        else:
            best = max(node.children, key=lambda s: s.q / s.n + c * np.sqrt(2 * np.log(node.n) / s.n))
            # scores = [s.q / s.n + c * np.sqrt(2 * np.log(node.n) / s.n) for s in node.children]
            # idx = np.argmax(scores)
            # self.route.append(idx)
            # best = node.children[idx]
            # if best.is_terminal:
            #     route = '_'.join(list(map(str, self.route))) + '_x'
            #     if route in self.terminal_routes:
            #         dummy = ''
        return best

    def default_policy(self, state):
        player_id = state.player()
        # ply_count = state.ply_count
        while not state.terminal_test():
            state = state.result(random.choice(state.actions()))
        win = -1 if state.utility(player_id) < 0 else 1
        # self.end_of_sim_stats = '{} {:>2}   {} {:>2}  {:>2}'.format(self.player_id, self.ply_count, player_id, ply_count, win)
        return -win

    def backup_negamax(self, node, reward):
        while node is not None:
            node.n += 1
            node.q += reward
            reward = -reward
            node = node.parent
        return node

    # UCT USING DICTIONARIES

    def uct_search_using_dict(self):
        # if self.state.terminal_test():
        #     print('0', self.state.terminal_test())
        #     return self.best_child(self.state, 0)

        # root_state = self.state
        if self.state not in self.context['mcts']:
            self.context['mcts'][self.state] = Stats()
        best_child = None
        for n in range(50):
            self.tree_policy_using_dict(self.state)

            # print(self.end_of_sim_stats, end=' ')
            # print([self.context['mcts'][root_state.result(a)] for a in self.context['mcts'][root_state].actions], end=' ')
            # print('_'.join(list(map(str, self.route))))
            # self.route = []

            # reward = self.default_policy(state)
            # self.backup_negamax(state, reward)
            best_child = self.best_child_using_dict(self.state, 0)
            # print(self.context['mcts'][root_state], best_child)

        # if self.state.result(best_child).terminal_test():
        #     print(best_child, n, self.state.ply_count)
        return best_child
        # return self.best_child(self.state, 0)

    def tree_policy_using_dict(self, state):
        if state.terminal_test():
            return state.utility(self.player_id)

        untried_actions = [a for a in state.actions() if a not in self.context['mcts'][state].actions]
        if untried_actions:
            state_next = self.expand_using_dict(state, random.choice(untried_actions))
            reward = -self.default_policy_using_dict(state_next)
            reward = self.backup_negamax_using_dict(state_next, reward)
        else:
            action = self.best_child_using_dict(state, c=1/np.sqrt(2))
            reward = self.tree_policy_using_dict(state.result(action))

        reward = self.backup_negamax_using_dict(state, reward)
        return reward

    def expand_using_dict(self, state, action):
        # self.route.append(len(self.context['mcts'][state].actions))
        self.context['mcts'][state].actions.append(action)
        state_next = state.result(action)
        if state_next not in self.context['mcts']:
            self.context['mcts'][state_next] = Stats(action)
        return state_next

    def best_child_using_dict(self, state, c=0):
        state0 = self.context['mcts'][state]
        child_states = [self.context['mcts'][state.result(a)] for a in state0.actions]
        if c == 0:
            best = max(child_states, key=lambda s: s.q / s.n)
        else:
            best = max(child_states, key=lambda s: s.q / s.n + c * np.sqrt(2 * np.log(state0.n) / s.n))
            # idx = np.argmax([s.q / s.n + c * np.sqrt(2 * np.log(state0.n) / s.n) for s in child_states])
            # self.route.append(idx)
            # best = child_states[idx]
        return best.parent

    def default_policy_using_dict(self, state):
        player_id = state.player()
        # ply_count = state.ply_count
        while not state.terminal_test():
            state = state.result(random.choice(state.actions()))
        win = -1 if state.utility(player_id) < 0 else 1
        # self.end_of_sim_stats = '{} {:>2}   {} {:>2}  {:>2}'.format(self.player_id, self.ply_count, player_id, ply_count, win)
        return win

    def backup_negamax_using_dict(self, state, reward):
        self.context['mcts'][state].n += 1
        self.context['mcts'][state].q += reward
        return -reward

    # EXTRAS

    def minimax(self, state, depth):

        def max_value(state, depth):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1))
            return value

        def min_value(state, depth):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1))
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))

    def minimax1(self, state, depth, is_max_player):
        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <= 0:
            return self.score(state)
        # if depth <= 0 or state.terminal_test():
        #     # return state.utility(self.player_id)
        #     return self.score(state)
        if is_max_player:
            best_value = float("-inf")
            for action in state.actions():
                best_value = max(best_value, self.minimax1(state.result(action), depth - 1, False))
            return best_value

        else:  # (* minimizing player *)
            best_value = float("inf")
            for action in state.actions():
                best_value = min(best_value, self.minimax1(state.result(action), depth - 1, True))
            return best_value

    def minimax2(self, state, depth, is_max_player):
        # if state.terminal_test():
        #     return state.utility(self.player_id)
        # if depth <= 0:
        #     return self.score(state)
        if depth <= 0 or state.terminal_test():
            return state.utility(self.player_id)

        best_value = float("-inf") if is_max_player else float("inf")
        opt = max if is_max_player else min
        for action in state.actions():
            best_value = opt(best_value, self.minimax2(state.result(action), depth - 1, not is_max_player))
        return best_value

    def alpha_beta(self, state, depth, alpha, beta, is_max_player):
        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <= 0:
            return self.score(state)

        # if depth <= 0 or state.terminal_test():
            # return self.score(state)
            # return state.utility(self.player_id)
        if is_max_player:
            v = float("-inf")
            for action in state.actions():
                v = max(v, self.alpha_beta(state.result(action), depth - 1, alpha, beta, False))
                # (* β cut-off *)
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v
        else:
            v = float("inf")
            for action in state.actions():
                v = min(v, self.alpha_beta(state.result(action), depth - 1, alpha, beta, True))
                # (* α cut-off *)
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

    def alpha_beta1(self, state, alpha, beta, is_max_player):
        if state.terminal_test():
            return state.utility(self.player_id)
        if is_max_player:
            v = float("-inf")
            for action in state.actions():
                v = max(v, self.alpha_beta1(state.result(action), alpha, beta, False))
                alpha = max(alpha, v)
                if beta <= alpha:
                    break  # (* β cut-off *)
            return v
        else:
            v = float("inf")
            for action in state.actions():
                v = min(v, self.alpha_beta1(state.result(action), alpha, beta, True))
                beta = min(beta, v)
                if beta <= alpha:
                    break  # (* α cut-off *)
            return v

    def alpha_beta2(self, state, depth, alpha, beta, is_max_player):
        # if state.terminal_test():
        #     return state.utility(self.player_id)
        # if depth <= 0:
        #     return self.score(state)

        if depth <= 0 or state.terminal_test():
            return self.score(state)
            # return state.utility(self.player_id)
        if is_max_player:
            alpha = float("-inf")
            for action in state.actions():
                alpha = max(alpha, self.alpha_beta2(state.result(action), depth - 1, alpha, beta, False))
                if alpha >= beta:
                    break  # (* β cut-off *)
            return alpha
        else:
            beta = float("inf")
            for action in state.actions():
                beta = min(beta, self.alpha_beta2(state.result(action), depth - 1, alpha, beta, True))
                if beta <= alpha:
                    break  # (* α cut-off *)
            return beta

    # MISC

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

    def greedy_score(self, state):
        own_loc = state.locs[self.player_id]
        own_liberties = state.liberties(own_loc)
        return len(own_liberties)

    def monte_carlo(self, node, reward):
        """
        A monte carlo update as in classical UCT.
        See feldman amd Domshlak (2014) for reference.
        """
        while node is not None:
            node.n += 1
            node.q = ((node.n - 1)/node.n) * node.q + 1/node.n * reward
            node = node.parent

