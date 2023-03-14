
# Evaluate simple Sympy expression
# Currently stuck at evaluating sympy expression, might make more sense to create simpler expressions and 
# evaluate them first
# Have taylor expansion function for symbolic nature, need to come up with nature of function generation
# Tokenise dataset using word to vector
# And then train and evaluate a LSTM and Transformer network. 
# Does not seem like the most challenging task in the world

from sympy import *
import matplotlib.pyplot as plt
import numpy as np

x = symbols("x")
expr = exp(x)

# So in general x really small it's decently accurate, but beyond that it's atrocious because the higher order terms start kicking in
def plot_taylor_function(function, x0, n):

    x = np.linspace(-10,10)
    taylor_rep = taylor(function, x0, n)
    print(taylor_rep.subs(0))
    
    y = [taylor_rep.subs(i) for i in x]
    plt.plot(x,y)
    plt.title(f'Taylor expansion of {function} to {n} order about {x0}')
    plt.show()
    

# Taylor approximation at x0 of the function 'function'
def taylor(function,x0,n):
    i = 0
    p = 0
    while i <= n:
        p = Add(p, (function.diff(x,i).subs(x,x0))/(factorial(i))*(x-x0)**i) # subs evaluates the function
        i += 1
    return p


function_operators = [cos(x), sin(x), exp(x), tan(x)]
# Create ranges of expressions using the three operators to Taylor expand and create dataset

function_dataset = []

for pow in range(0, 6):
    for i in range(0, len(function_operators)):
        for j in range(0, len(function_operators)):
            function_dataset.append(function_operators[i]**pow * function_operators[j])
            function_dataset.append(function_operators[i] * function_operators[j]**pow)



taylor_rep = [taylor(i,0,4) for i in function_dataset]

# Write in text file 
with open('taylor.txt', 'w') as f:
    for i in range(0, len(function_dataset)):
        f.write(f'{function_dataset[i]}={taylor_rep[i]}\n')

f.close()








