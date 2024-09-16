from tytan import symbols, symbols_list, Compile
import numpy as np
import neal
import optuna
from matplotlib import pyplot as plt

def A(n):
    A = np.array([], np.int64)
    while True:
        A = np.append(A, 2**(int(np.log2(n + 1)) - 1))
        n -= A[-1]
        if n == 0:
            break
    return A

def add_qubo(qubo, key, var):
    if key in qubo:
        qubo[key] += var
    else:
        qubo[key] = var


# 定数

S = 2000
N = 36

m = 0
M = 70

# 各看板の座標を生成

cod = np.random.rand(S, 2)

# 各営業マンの住所の座標を生成

add = np.zeros((0, 2))

for i in range(6):
    for j in range(6):
        add = np.append(add, [[0.2 * i, 0.2 * j]], axis=0)

# それぞれの営業マンが行くべき看板の番号
Sn = []

# for p in cod:
#     plt.plot(p[0], p[1], marker=".", color="red")
# for p in add:
#     plt.plot(p[0], p[1], marker=".", color="black")
# plt.show()

# 距離函数

def dist(x, y):
    return float(np.sum(np.abs(x - y)))

# 営業マンが訪問可能な看板までの距離
r = 0.3

# 各営業マンが訪問可能な看板の番号
visitable = []
visitable_N = []

for n in range(N):
    visitable.append([])
    for s in range(S):
        if dist(add[n], cod[s]) <= r:
            visitable[-1].append(s)
    visitable_N.append(len(visitable[-1]))

print(np.min(visitable_N))

visitable_S = []
visitable_S_N = []
for s in range(S):
    visitable_S.append([])
    for n in range(N):
        if dist(add[n], cod[s]) <= r:
            visitable_S[-1].append(n)
    visitable_S_N.append(len(visitable_S[-1]))
print(np.min(visitable_S_N))

# 第一段　クラスタリング


def objective(trial):

    l = [0.001, [trial.suggest_float("l_1", 0, 10), trial.suggest_float("l_2", 0, 100)]]

    # 変数

    x = [[symbols("x{}_{}".format(n, s)) for s in visitable[n]] for n in range(N)]

    # 制約条件

    H = 0

    aux_1 = []
    aux_2 = []

    for n in range(N):
        # aux_coef = A(len(visitable[n]) - m)
        # aux_coef_len = len(aux_coef)
        # aux_1.append(symbols_list(aux_coef_len, "aux{}".format(n) + "_{}"))

        # H += l[1][0] * (m + sum(aux_coef[i] * aux_1[-1][i] for i in range(aux_coef_len)) - sum(x[n][i] for i in range(len(visitable[n]))))**2


        aux_coef = A(M)
        aux_coef_len = len(aux_coef)
        aux_2.append(symbols_list(aux_coef_len, "aux{}".format(n) + "_{}"))

        H += l[1][0] * (sum(x[n][s] for s in range(len(visitable[n]))) + sum(aux_coef[i] * aux_2[-1][i] for i in range(aux_coef_len)) - M)**2

    for s in range(S):
        H += l[1][1] * (sum(x[n][i] for n in range(N) for i in range(len(visitable[n])) if s == visitable[n][i]) - 1)**2

    qubo, offset = Compile(H).get_qubo()

    print("Get Qubo")

    # 目的函数

    H = 0

    for n in range(N):
        for i in range(len(visitable[n])):
            for i_ in range(i + 1, len(visitable[n])):
                add_qubo(qubo, ("x{}_{}".format(n, visitable[n][i]), "x{}_{}".format(n, visitable[n][i_])), l[0] * dist(cod[visitable[n][i]], cod[visitable[n][i_]]))

    solver = neal.SimulatedAnnealingSampler()

    x = solver.sample_qubo(Q=qubo).first.sample

    # 制約条件を充たすか確認

    Error_num = 0

    # 1

    for n in range(N):
        S_x = 0
        for s in visitable[n]:
            S_x += int(x["x{}_{}".format(n, s)])
        if S_x > M:
            Error_num += 1

    # 2

    S_n_list = []
    
    for s in range(S):
        S_n = 0
        for n in range(N):
            if s in visitable[n]:
                S_n += int(x["x{}_{}".format(n, s)])
        S_n_list.append(S_n)
        if S_n != 1:
            Error_num += 1
    
    if Error_num == 0:
        global Sn
        for n in range(N):
            Sn.append([])
            for s in visitable[n]:
                if x["x{}_{}".format(n, s)] == 1:
                    Sn[-1].append(s)
        trial.study.stop()

    return Error_num

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1000)

#　第二段　各営業マンに対して巡回セールスマンを解く

print("\n\n")

# それぞれの営業マンが回る順番に看板の番号を並べる
turn_n = []

def objective(n, Sn_n, add_n, cod_n):
    def objective_n(trial):
        l = [0.001, [trial.suggest_float("l_1", 0, 100), trial.suggest_float("l_2", 0, 100)]]

        N_n = len(cod_n)
        x_tp = symbols_list((N_n, N_n), "x{}_{}")

        H = 0

        # 目的函数

        for s in range(N_n):
             H += l[0] * (dist(add_n, cod_n[s])  * (x_tp[0][s] + x_tp[N_n - 1][s]))

        for t in range(1, N_n - 1):
            for s in range(N_n):
                for s_ in range(s + 1, N_n):
                    H += l[1][0] * dist(cod_n[s], cod_n[s_]) * x_tp[t][s] * x_tp[t + 1][s_]
                   
        # 制約条件

        for s in range(N_n):
            H += l[1][0] * (sum(x_tp[t][s] for t in range(N_n)) - 1)**2

        for t in range(N_n):
            H += l[1][1] * (sum(x_tp[t][s] for s in range(N_n)) - 1)**2

        qubo, offset = Compile(H).get_qubo()

        print("Get qubo")

        solver = neal.SimulatedAnnealingSampler()

        x = solver.sample_qubo(Q=qubo).first.sample

        # 制約条件を充たすか確認

        Error_num = 0

        for n in range(N_n):
            if sum(x["x{}_{}".format(t, s)] for t in range(N_n)) != 1:
                Error_num += 1
        
        for t in range(N_n):
            if sum(x["x{}_{}".format(t, s)] for s in range(N_n)) != 1:
                Error_num += 1
        
        if Error_num == 0:
            global turn_n
            for t in range(N_n):
                for s in range(N_n):
                    if x["x{}_{}".format(t, s)] == 1:
                        turn_n.append(Sn_n[s])

            trial.study.stop()
        
        return Error_num

    return objective_n

for n in range(N):
    print("営業マン {}".format(n))

    Sn_n = Sn[n]
    add_n = add[n]
    cod_n = cod[Sn_n]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective(n, Sn_n, add_n, cod_n), n_trials=1000)

print(turn_n)

for n in range(N):
    color = list(np.random.choice(range(256), 3))
    
    plt.plot(add[n], turn_n[0], color)
    plt.plot(add[n], turn_n[len(turn_n) - 1], color)

    for s in len(turn_n) - 1:
        plt.plot(turn_n[s], turn_n[s + 1], color)

plt.show()
