```

```

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

The fist component of input is the name of the variable. The second is the minimum value and the last is the maximum value, so this variable "x" takes $2,3,4,5$​​. 

 `Formula` is a class whose instance is generated automatically when variables are operated. For example, 

```
f = - x[0]**2 -  3 * x[1]
g = x[0] > x[1]
```

then `f`,  `g` are instances of `Formula`. Finally, we can define a problem using `Problem`.

```
P = Problem() 
P.add_objective(f)
P.add_constraint(g)
```

 Finally, we can get QUBO by `compile` method.

```
qubo = P.compile([10])

sampler = neal.SimulatedAnnealingSampler()
result = sampler.sample_qubo(qubo.todict()).first.sample
solution = P.solution(result, "neal")
print(solution)
```

where the input of `compile` method of `Problem` class  is the weights vector of the added constraints. The solution is almost always below .

```
({'x0': 5, 'x1': 4}, [True])
```

The second component means whether each solution satisfies each constraint conditions. In the above case, because $5 > 4$, the return is true. Note that heuristic algorithms do not necessarily return an exact solution, so we always need to pay attention to the second component. 

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

P = quboassist.Problem()

f = - x[0]**2 - x[1]
P.add_objective(f)

g = x[0] > x[1]
P.add_constraint(g)

h = x[0] + x[1] == x[2]
P.add_constraint(h)

sampler = neal.SimulatedAnnealingSampler()

def objective(trial):
    
    w = [trial.suggest_float("w{}".format(i), 0, 5) for i in range(2)  
    qubo = P.compile(w)

    result = sampler.sample_qubo(qubo.todict()).first
    solution = P.solution(result.sample)

    print("\n")
    print(solution)

    obj = w[0] + w[1] + 10 * sum(np.logical_not(solution[1]))
    val = result.energy
    
    # Note that result.energy and the value of objective function may differ by a constant which appears when expanding the product of variables!

    global best_solution, best_val
    
    if np.all(solution[1]) and val < best_val:
        best_val = val
        best_solution = copy(solution)
    
    return obj

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("\n\nBest Solutuion")
print(best_solution)
```



## What kind of technique is used?



The core is as below. We leave it to the reader to consider how to use this to represent any bounded integer variable as a linear combination of a constant and a binary variable, and to convert an inequality into an equality.



Algorithm

------

Input: $n$

Output: $A(n)$

​        Define $A_0(n) \leftarrow 2^{\lfloor \log_2 (n + 1) \rfloor - 1}, n_1 \leftarrow n - A_0(n), k \leftarrow 0$

​        while $n_{k+1} \neq 0$ do

​                $A_{k + 1}(n) \leftarrow 2^{\lfloor \log_2 (n_{k + 1} + 1) \rfloor - 1}$	

​                $k \leftarrow k + 1$

​                $n_{k + 1}\leftarrow n_k - A_k(n)$

​        end while

------



*Lemma*

For all $n \in \mathbb N$, the sequence $A(n)$​ is finite and the length is at most


$$
2\lfloor\log_2(n +1)\rfloor.
$$


Moreover, a function $f: [0,1]^k \rightarrow \mathbb N$ defined as:


$$
f(x_1,x_2,...,x_k):= \sum_{i =1}^k A_i(n) x_i
$$


takes the all numbers $0,...,n$ and no other values.



*Proof.*

Since $A_0(n)$ is the only power of two which satisfies $4A_0(n)> n+1 \geq 2 A_0(n)$, 


$$
n - A_0(n) \geq A_0(n)-1
$$


i.e.


$$
A_1(n)=2^{\lfloor\log_2(n_1 +1)\rfloor-1}\geq \frac{1}{2}A_0(n)
$$


holds if $n_1 \neq 0$. Therefore if $A_0(n) \geq 2$, then $A_1(n) = A_0(n)$ or $A_1(n) =\frac{1}{2} A_0(n)$. Moreover, in the case that the first two numbers of $A(n)$​ is same, 


$$
n_2=n-2A_1(n)<2A_1(n)-1
$$


i.e.


$$
A_2(n)=2^{\lfloor\log_2(n_2 +1)\rfloor-1}<A_1(n).
$$


Hence the same number appears at most two times in $A(n)$ and the exponent $\lfloor\log_2(n_k +1)\rfloor-1$ is monotonically non-increasing, thus the length of $A(n)$ is at most


$$
2\lfloor\log_2(n +1)\rfloor,
$$


moreover by the same reason, we also conclude the sequence $A(n)$ includes all powers of two
less than or equal to


$$
2^{\lfloor\log_2(n +1)\rfloor-1}(= A_0(n)).
$$


Therefore, numbers $0,...,2A_0(n)-1$​ can be expressed as


$$
 \sum_{i=1}^kA_i(n)x_i
$$


and numbers $n-2A_0(n),...,n$​ can be expressed as


$$
n-\sum_{i=1}^kA_i(n)y_i=\sum_{i=1}^kA_i(n)(1-y_i).
$$


Since


$$
n-2A_0(n)< 2A_0(n) -1,
$$


finally all numbers $0,...,n$​​ can be expressed as


$$
\sum_{i=1}^kA_i(n)x_i.
$$

<div style="text-align: right;">
□
</div>
