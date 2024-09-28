# quboassist

This is a package to generate QUBO which can be input to dwave-neal simulated annealing solver and reconstruct the solution of the original problem from the output.

## What problems is it applicable to?

This package converts the problem below into QUBO form which can be input directly to dwave-neal package:


$$
{\rm minimize}\ \ \ f(x) =x^t Ax\\
$$

$$
s.t. \ \ \ \forall i, a_i \leq x_i\leq b_i,x_i\in\mathbb Z
$$

$$
\forall j, \sum_ kc_{jk} x_k \geq d_j\ \ (c_{jk}, d_j \in \mathbb Z)
$$



where $A$ is a symmetric real matrix. I.e. problems where the objective function is quadratic, all variables are bounded and integer, all constraints are linear and their all coefficients are integer.

## How  to use

First, import the all classes of quboassist.

```
from quboassist import *
import neal
```

There are three classes: `Variable`, `Formula`, `Problem`. Using `Variable` class, we can define variables.

```
x = [Variable("x{}".format(i), 2, 5) for i in range(10)]
```

The fist component of input is the name of the variable. The second is the minimum value and the last is the maximum value, so this variable "x" takes $2,3,4,5$​.  If you want to change the passible values of the variable, you can do the following.

```
x[4].change_max(4)
x[5].change_min(-2)
```

`Formula` is a class whose instance is generated automatically when variables are operated. For example, 

```
f = - x[0]**2 -  3 * x[1]
g = x[0] > x[1]
```

then `f`,  `g` are instances of `Formula`. Finally, we can define a problem using `Problem`.

```
P = Problem() 
P.add_objective(f)
P.add_constraint(10, g)
```

where the first input of `add_constraint` method is the weight of the constraint. Finally, we can get QUBO by `compile` method.

```
P.compile()

sampler = neal.SimulatedAnnealingSampler()
result = sampler.sample_qubo(P.qubo).first.sample
solution = P.solution(result)
print(solution)
```

The solution is almost always below .

```
({'x0': 5, 'x1': 4}, [True])
```

The second component means whether the solution satisfies each constraint conditions. In the above case, because $5 > 4$, the return is true. Note that heuristic algorithms do not necessarily return an exact solution, so we always need to pay attention to the second component. 

In general, increasing the weight $w_i$ tends to make it easier to satisfy the condition, but the objective function becomes relatively smaller. Therefore we propose to use a library called optuna to tune these hyperparameters $w_i$.

A sample code is showed below.

```
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
```

# What kind of technique is used?



The core is as below:



*Lemma*



