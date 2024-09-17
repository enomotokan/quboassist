from tytan import symbols, symbols_list, Compile
import numpy as np
import neal
import optuna
from matplotlib import pyplot as plt, colors
import random

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

def choose_colors(num_colors):
  # matplotlib.colors.CSS4_COLORSの値だけをリストにする
  tmp = list(colors.CSS4_COLORS.values())
  # リストにしたものをランダムにシャッフルする
  random.shuffle(tmp)
  # 必要な数だけ先頭から取り出してreturnする
  label2color = tmp[:num_colors]
  return label2color

# 定数

S = 2000
N = 36

m = 40
M = 60

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
    return float(np.linalg.norm(x - y))

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

dist_list = []
dist_order_arg_n_s = []
dist_order_arg_s_n = []
S_n = [set() for n in range(N)]
S_n_len = [0 for n in range(N)]

for s in range(S):
    dist_list.append([])
    for n in range(N):
        dist_list[-1].append(dist(add[n], cod[s]))
    dist_order_arg_s_n.append(np.argsort(dist_list[-1]))
    S_n[dist_order_arg_s_n[-1][0]].add(s)
    S_n_len[dist_order_arg_s_n[-1][0]] += 1

dist_list = np.array(dist_list)

for i in range(100000):
    S_n_over = []
    for n in range(N):
        dist_order_arg_n_s.append(np.argsort(dist_list[n]))
        if S_n_len[n] > int(S / N) * 1.4:
            S_n_over.append(n)
    n = S_n_over[np.random.randint(len(S_n_over))]
    s = list(S_n[n])[np.argsort(dist_list[:, n][list(S_n[n])])[-1]]

    rand = np.random.random()

    n_cand = []
    for n_ in dist_order_arg_s_n[s]:
        if n_ != n:
            n_cand.append(n_)
        if len(n_cand) == 2:
            break
    if rand < 1 / (1 + np.exp(-0.0001 * i)):
        n_cand = n_cand[0]
    else:
        n_cand = n_cand[1]
    
    S_n[n].remove(s)
    S_n[n_cand].add(s)

    S_n_len[n] -= 1
    S_n_len[n_cand] += 1

color = choose_colors(N)

print(S_n_len)

for n in range(N):
    for s in list(S_n[n]):
        plt.scatter(cod[s][0], cod[s][1], c=color[n])
plt.show()

#　第二段　各営業マンに対して巡回セールスマンを解く

print("\n")

# それぞれの営業マンが回る順番に看板の番号を並べる
turn_n = []

def objective(n, Sn_n, add_n, cod_n):
    def objective_n(trial):
        l = [0.001, [trial.suggest_float("l_1", 0, 100), trial.suggest_float("l_2", 0, 100)]]

        N_n = len(cod_n)
        qubo = {}

        x_tp = symbols_list((N_n, N_n), "x{}_{}")

        H = 0

        # 目的函数

        for s in range(N_n):
             add_qubo(qubo, ("x{}_{}".format(0, s), "x{}_{}".format(0, s)), l[0] * (dist(add_n, cod_n[s])))
             add_qubo(qubo, ("x{}_{}".format(N_n - 1, s), "x{}_{}".format(N_n - 1, s)), l[0] * (dist(add_n, cod_n[s])))

        for t in range(1, N_n - 1):
            for s in range(N_n):
                for s_ in range(N_n):
                    add_qubo(qubo, ("x{}_{}".format(t, s), "x{}_{}".format(t + 1, s_)), l[0] * dist(cod_n[s], cod_n[s_]))
                   
        # 制約条件

        for s in range(N_n):
            for t in range(N_n):
                for t_ in range(t, N_n):
                    if t == t_:
                        add_qubo(qubo, ("x{}_{}".format(t, s), "x{}_{}".format(t, s)), - l[1][0])
                    else:
                        add_qubo(qubo, ("x{}_{}".format(t, s), "x{}_{}".format(t_, s)), 2 * l[1][0])

        for t in range(N_n):
            for s in range(N_n):
                for s_ in range(N_n):
                    if s == s_:
                        add_qubo(qubo, ("x{}_{}".format(t, s), "x{}_{}".format(t, s)), - l[1][1])
                    else:
                        add_qubo(qubo, ("x{}_{}".format(t, s), "x{}_{}".format(t, s_)), 2 * l[1][1])

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
    print("\n営業マン {}".format(n))

    Sn_n = list(S_n[n])
    add_n = add[n]
    cod_n = cod[Sn_n]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective(n, Sn_n, add_n, cod_n), n_trials=1000)

for n in range(N):
    plt.plot(add[n], cod[turn_n[0]], color[n])
    plt.plot(add[n], cod[turn_n[len(turn_n) - 1]], color[n])

    for s in range(len(turn_n) - 1):
        plt.plot(cod[turn_n[s]], cod[turn_n[s + 1]], color[n])

plt.show()
