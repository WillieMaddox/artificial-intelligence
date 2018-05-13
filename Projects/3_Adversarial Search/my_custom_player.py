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
        self.max_book_depth = 2
        if self.data is None:
            self.book = defaultdict(Counter)
        else:
            self.book = self.data
        self.context = self.book

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
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        # self.queue.put(random.choice(state.actions()))
        # print('A', state.player(), state.board, state.actions())
        self.book = self.context
        self.build_tree(state, self.max_book_depth)
        self.book = {k: max(v, key=v.get) for k, v in self.book.items()}
        print(self.book)
        if state.terminal_test():
            self.save(self.book)

        self.queue.put(self.book[state.board])
        # print('B', state.player(), state.board, state.actions())
        self.context = self.book  # self.context will contain this object on the next turn

        # Iterative Deepening
        # best_move = None
        # for depth in range(1, depth_limit+1):
        #     best_move = minimax_decision(gameState, depth)
        # return best_move

    def save(self, book):
        with open("data.pickle", 'wb') as ofs:
            print(book)
            pickle.dump(book, ofs)

    def build_tree(self, state, depth=2):
        if depth <= 0 or state.terminal_test():
            return -self.simulate(state)
        action = random.choice(state.actions())
        reward = self.build_tree(state.result(action), depth - 1)
        print(self.book)
        self.book[state.board][action] += reward
        return -reward

    def simulate(self, state):
        player_id = state.player()
        while not state.terminal_test():
            state = state.result(random.choice(state.actions()))
        return -1 if state.utility(player_id) < 0 else 1