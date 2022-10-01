import numpy as np
import itertools

class easyGame():
    def __init__(self, values, func):
        self.num_player = len(values)
        self.values = np.array(values)
        self.func = func
        
    def evaluate(self, subset):
        return self.func(np.dot(self.values, subset))

def shapley_value(game, modified=False):
    num_player = game.num_player
    shapley_value = np.zeros(num_player)
    in_subset = np.zeros(num_player)
    for subset in itertools.product([1,0], repeat=num_player):
        subset = np.array(subset)
        r = np.sum(subset)
        if r:
            index = subset==1
            coef = 0
            for subsubset in itertools.product([1,0], repeat=r):
                t = np.sum(subsubset)
                in_subset[index] = subsubset
                coef += (-1)**(r-t) * game.evaluate(in_subset)
            if modified:
                shapley_value += (coef*r) * subset
            else:
                shapley_value += (coef/r) * subset                
            in_subset.fill(0)
    shapley_value[shapley_value<1e-12] = 0
    return shapley_value

def test_efficiency(game, modified=False):
    print("Testing efficiency")
    num_player = game.num_player
    v = shapley_value(game, modified)
    expected_sum = game.value(np.ones(num_player)) - game.value(np.zeros(num_player))
    msg = """
    The expected sum is {},
    and the sum of the calculated values is {}.
    """
    print(msg.format(expected_sum, np.sum(v)))
    
    
def test_symmetry(game, modified=False):
    print("Testing symmetry...")
    pi = np.random.permutation(game.num_player)
    v = shapley_value(game, modified)
    game.values = game.values[pi]
    v_perm = shapley_value(game, modified)
    msg = """
    Permuting the game, and the calculated values are
    {}
    Permuting the calculated Shapley values, the result is 
    {}
    """
    print(msg.format(v_perm, v[pi]))
    
def test_linearity(values, f=lambda x:x**(1/3), g=np.sin, modified=False):
    print("Testing linearity...")
    game_f = easyGame(values, f)
    game_g = easyGame(values, g)
    game_sum = easyGame(values, lambda x:f(x)+g(x))
    vf = shapley_value(game_f, modified)
    vg = shapley_value(game_g, modified)
    v_sum = shapley_value(game_sum, modified)
    msg = """
    The sum of the values generated from the two games is
    {}
    The generated values for the sum game is
    {}
    """
    print(msg.format(vf+vg, v_sum))