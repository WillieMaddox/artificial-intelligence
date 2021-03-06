import sys
import random
from math import sqrt, log
from sample_players import DataPlayer


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

    @property
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self):
        return len(self.actions) == 0


class CustomPlayer(DataPlayer):
    def __init__(self, player_id):
        super().__init__(player_id)
        self.data = {} if self.data is None else self.data
        # print(sys.getsizeof(self.data))
        self.context = {}

    def get_action(self, state):
        if state in self.data:
            self.queue.put(self.data[state])
        else:
            # RANDOM
            # self.queue.put(random.choice(state.actions()))

            # GREEDY
            # self.queue.put(max(state.actions(), key=lambda x: self.greedy_score(state.result(x))))

            # MINIMAX
            # self.queue.put(self.minimax(state, depth))
            # self.queue.put(max(state.actions(), key=lambda x: self.minimax1(state.result(x), depth - 1, False)))
            # self.queue.put(max(state.actions(), key=lambda x: self.minimax2(state.result(x), depth - 1, False)))

            # ALPHA-BETA
            # self.queue.put(max(state.actions(), key=lambda x: self.alpha_beta(state.result(x), depth - 1, float("-inf"), float("inf"), False)))
            # self.queue.put(max(state.actions(), key=lambda x: self.alpha_beta1(state.result(x), float("-inf"), float("inf"), False)))
            # self.queue.put(max(state.actions(), key=lambda x: self.alpha_beta2(state.result(x), depth - 1, float("-inf"), float("inf"), False)))

            # MCTS-UCT
            self.queue.put(self.uct_search(state))

    def uct_search(self, state):
        root_node = Node(state)
        for n in range(100):
            node = self.tree_policy(root_node)
            reward = self.default_policy(node.state)
            self.backup_negamax(node, reward)
        return self.best_child(root_node, 0).action

    def tree_policy(self, node):
        while not node.is_terminal:
            if not node.is_fully_expanded:
                return self.expand(node)
            else:
                node = self.best_child(node)
        return node

    def expand(self, node):
        action = node.untried_actions.pop()
        child = Node(node.state.result(action), action, node)
        node.children.append(child)
        return child

    def best_child(self, node, c=1/sqrt(2)):
        if c == 0:
            best = max(node.children, key=lambda s: s.q / s.n)
        else:
            best = max(node.children, key=lambda s: s.q / s.n + c * sqrt(2 * log(node.n) / s.n))
        return best

    def default_policy(self, state):
        player_id = state.player()
        while not state.terminal_test():
            state = state.result(random.choice(state.actions()))
        win = -1 if state.utility(player_id) < 0 else 1
        return -win

    def backup_negamax(self, node, reward):
        while node is not None:
            node.n += 1
            node.q += reward
            reward = -reward
            node = node.parent
        return node

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

