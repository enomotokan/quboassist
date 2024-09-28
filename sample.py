import neal
import quboassist

x = [quboassist.Variable("x{}".format(i), 2, 5) for i in range(10)]
P = quboassist.Problem()

f = x[0]**2 + 3 * x[1]
P.add_objective(f)

g = x[0] > 2 * x[1]
P.add_constraint(5, g)

P.compile()
sampler = neal.SimulatedAnnealingSampler()
result = sampler.sample_qubo(P.qubo).first.sample
solution = P.solution(result)

print(solution)