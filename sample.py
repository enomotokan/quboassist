import neal
import quboassist
import optuna
import numpy as np
from copy import copy

x = [quboassist.Variable("x{}".format(i), 0, 3) for i in range(10)]

best_solution = []
best_val = np.inf

def objective(trial):

    P = quboassist.Problem()

    f = - x[0]**2 - x[1]
    P.add_objective(f)

    w = [trial.suggest_float("w{}".format(i), 0, 5) for i in range(2)]

    g = x[0] > x[1]
    P.add_constraint(w[0], g)

    h = x[0] + x[1] == x[2]
    P.add_constraint(w[1], h)

    P.compile()

    sampler = neal.SimulatedAnnealingSampler()
    result = sampler.sample_qubo(P.qubo).first
    solution = P.solution(result.sample)
    
    obj = w[0] + w[1] + 10 * sum(np.logical_not(solution[1]))
    val = result.energy
    
    # Note that result.energy and the value of objective function may differ by a constant which appears when expanding the product of variables!

    global best_solution, best_val
    
    if val < best_val:
        best_val = val
        best_solution = copy(solution)
    
    return obj

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

print(best_solution)