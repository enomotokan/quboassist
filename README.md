# quboassist



This is a package to generate QUBO which can be input to dwave-neal simulated annealing solver and reconstruct the solution of the original problem.

# What problems is it applicable to?

This package converts the problem below into QUBO form which can be input directly to dwave-neal package:
$$
{\rm minimize}\ \ \ f(x) =x^t Ax\\
s.t. \ \ \ \forall i, a_i \leq x_i\leq b_i,x_i\in\mathbb Z
$$
i.e. a problem where the objective function is quadratic and all variables are bounded and integer.

# How  to use

First, import the all classes of quboassist.

```
from quboassist import *
import neal
```

There are three classes: `Variable`, `Formula`, `Problem`. Using `Variable` class, we can define variables.

```
x = [Variables("x{}".format(i), 2, 5) for i in range(10)]
```

The fist component of input is the name of the variable. The second is the minimum value and the last is the maximum value, so this variable "x" takes $2,3,4,5$​.

Formula is a class which is generated automatically when variables are operated. For example, 

```
f = x[0]**2 + 3 * x[1]
g = x[0] > 5 * x[1]
```

Then `f`,  `g` are instances of `Formula`. Finally, we can define a problem using `Problem`.

```
P = Problem() 
Problem.add_objective(f)
Problem.add_constraint(10, g)
```

where the first input of `add_constraint` method is the weight of the constraint. Finally, we can get QUBO by `compile` method.

```
P.compile()

sampler = neal.SimulatedAnnealingSampler()
result = sampler.sample_qubo(P.qubo).first.sample
solution = P.solution(result)
print(solution)
```

