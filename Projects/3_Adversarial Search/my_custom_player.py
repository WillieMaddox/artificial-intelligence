import pickle
import random
from collections import defaultdict, Counter
from sample_players import DataPlayer


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
        self.max_book_depth = 4
        if self.data is None:
            book = defaultdict(Counter)
            best = {}
        else:
            book = self.data
            best = {k: max(v, key=v.get) for k, v in self.data.items()}

        self.context = {
            'book': book,
            'best': best
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
        depth = state.ply_count
        book = self.context['book']
        state_hash = self.get_state_hash(state)
        # print('state    ', state)
        # print('book     ', book)
        if depth < self.max_book_depth:
            # print('\n*************** build tree ***************')
            self.build_tree(state, book, self.max_book_depth - depth)
            # print('SAVING BOOK', len(self.context['book']))
            self.save(book)
            # print('******************************************\n')

        if random.random() < 0.25 and state_hash in self.context['best']:
            print('TAKE GOOD ACTION')
            action = self.context['best'][state_hash]
            print('(board, loc, Player), depth, action', state_hash, depth, action)
        else:
            # print('TAKE RANDOM ACTION')
            action = random.choice(state.actions())

        self.queue.put(action)
        # self.queue.put(self.book[state.board])
        # Iterative Deepening
        # best_move = None
        # for depth in range(1, depth_limit+1):
        #     best_move = minimax_decision(gameState, depth)
        # return best_move
        # if self.is_terminal(state, action):
        #     print('SAVING BOOK')
        #     self.save(book)
        # if state.terminal_test():
        #     print('SAVING BOOK')
        #     self.save(book)

    def build_tree(self, state, book, depth=4):
        if depth <= 0 or state.terminal_test():
            return -self.simulate(state)
        action = random.choice(state.actions())
        # print('depth', depth, 'action', action)

        reward = self.build_tree(state.result(action), book, depth - 1)
        state_hash = self.get_state_hash(state)
        book[state_hash][action] += reward
        # print('reward {:2} book {}'.format(depth, reward, book))

        return -reward

    def simulate(self, state):
        player_id = state.player()
        while not state.terminal_test():
            action = random.choice(state.actions())
            # print(action)
            state = state.result(action)
            # print(state)
        return -1 if state.utility(player_id) < 0 else 1

    def get_state_hash(self, state):
        return state.board, state.locs, state.player()

    def is_terminal(self, state, action):
        state = state.result(action)
        return state.terminal_test()

    def save(self, book):
        with open("data.pickle", 'wb') as ofs:
            # print(book)
            pickle.dump(book, ofs)

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

    def utc_search(self, state):
        pass

    def tree_policy(self):
        pass