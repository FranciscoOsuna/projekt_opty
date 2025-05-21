import numpy as np
import hooke_jeeves as hj
import generalized_search as gs
import problems
import graph

optimizer = gs.GeneralizedPatternSearch(problems.three_hump_camel, x0 = [-5, -5], step_size=0.5)
solution = optimizer.optimize()

print(solution)
graph.visualise(problems.three_hump_camel, point=solution)
graph.animate(problems.three_hump_camel, optimizer.history)
