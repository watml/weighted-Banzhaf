from Tools.ShapleyValue import *

print("Testing if the Shapley value is just a multiple of its modified version, which only satisfies Dummy, Symmetry and Linearity")

# Each value is assigned to one player to create a gamem, each value has to be nonnegative
values = [1,2,0,3,4,0,0,2,2,0]
# The setting here makes that a player is dummy iff it is with value 0
game = easyGame(values, lambda x:x**(1/3))

# caluculate values using Shapley value method
v = shapley_value(game)
# calculate using the modified version
vm = shapley_value(game, True)

# scale vm to v based on the first component
vm = vm * v[0]/vm[0]
msg = """
Result from Shapley value method is
{}
Scaling the values generated from the modified version so that they match on the 1st component yields
{}
"""
print(msg.format(v,vm))

# below is used to test symmetry for the modified method
#test_symmetry(game, True)

# below is used to test linearity for the modified method
#test_linearity(values, modified=True)  
