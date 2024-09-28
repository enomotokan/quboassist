import neal
import quboassist

x = [quboassist.Variable("x{}".format(i), 0, 3) for i in range(10)]
P = quboassist.Problem()

f = - x[0]**2 - x[1]
P.add_objective(f)

g = x[0] > x[1]
P.add_constraint(5, g)

h = x[0] + x[1] == x[2]
P.add_constraint(5, h)

P.compile()
sampler = neal.SimulatedAnnealingSampler()
result = sampler.sample_qubo(P.qubo).first.sample
solution = P.solution(result)

print(solution)