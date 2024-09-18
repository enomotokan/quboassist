from tytan import symbols, symbols_list, Compile
import numpy as np
import neal
import optuna
from matplotlib import pyplot as plt, colors, collections as mc
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

color = choose_colors(N)

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

# 第一段　クラスタリング

n_cand = [[] for s in range(S)]
dist_list = []

for n in range(N):
    dist_list.append([])
    for s in range(S):
        dist_list[-1].append(dist(add[n], cod[s]))

    r = 0.01

    while True:
        dist_n_r = np.array(dist_list[n]) <= r
        if sum(dist_n_r) > int(S / N) * 3:
            for s in np.arange(S)[dist_n_r]:
                n_cand[s].append(n)
            break
        r += 0.01

dist_list = np.array(dist_list)
S_n = [set() for n in range(N)]
S_n_len = np.zeros(N)

for s in range(S):
    if len(n_cand[s]) >= 2:
        n = n_cand[s][np.argmin(S_n_len[n_cand[s]])]

    elif len(n_cand[s]) == 1:
        n = n_cand[s][0]

    else:
        n = np.argmin(dist_list[:, s])
    
    S_n[n].add(s)
    S_n_len[n] += 1

# 遠い看板同士を交換

near_n_list = [[] for n in range(N)]

for n in range(N):
    for n_ in range(n + 1, N):
        if dist(add[n], add[n_]) <= 0.3:
            near_n_list[n].append(n_)
            near_n_list[n_].append(n)

iter = 1000000
for i in range(iter):
    n = np.random.randint(N)
    n_ = np.random.choice(near_n_list[n])

    # if i < iter * 0.7:
    #     s = random.choice(np.array(list(S_n[n]))[np.where(dist_list[n][list(S_n[n])] > np.mean(dist_list[n][list(S_n[n])]))[0]])
    #     s_ = random.choice(np.array(list(S_n[n_]))[np.where(dist_list[n_][list(S_n[n_])] > np.mean(dist_list[n_][list(S_n[n_])]))[0]])
    # else:
    # s = random.choice(np.array(list(S_n[n]))[np.argsort(-dist_list[n][list(S_n[n])])[:10]])
    # s_ = random.choice(np.array(list(S_n[n_]))[np.where(dist_list[n_][list(S_n[n_])] > np.mean(dist_list[n_][list(S_n[n_])]))[0]])
    if i < iter * 0.7:
        s = random.choice(np.array(list(S_n[n]))[np.argsort(dist_list[n_][list(S_n[n])])[:20]])
        s_ = random.choice(np.array(list(S_n[n_]))[np.argsort(dist_list[n][list(S_n[n_])])[:20]])
    else:
        s = random.choice(np.array(list(S_n[n]))[np.argsort(-dist_list[n][list(S_n[n])])[:20]])
        s_ = random.choice(np.array(list(S_n[n_]))[np.where(dist_list[n_][list(S_n[n_])] > np.mean(dist_list[n_][list(S_n[n_])]))[0]])
    
    
    if dist_list[n, s] + dist_list[n_, s_] > dist_list[n, s_] + dist_list[n_, s]:
        S_n[n].remove(s)
        S_n[n].add(s_)
    
        S_n[n_].remove(s_)
        S_n[n_].add(s)

print(S_n_len)

for n in range(N):
    for s in list(S_n[n]):
        plt.scatter(cod[s][0], cod[s][1], c=color[n])
plt.show()

#　第二段　各営業マンに対して巡回セールスマンを解く

print("\n")

# それぞれの営業マンが回る順番に看板の番号を並べる
turn_n = [[] for n in range(N)]

value = np.inf

def objective(n, Sn_n, add_n, cod_n):

    def objective_n(trial):
        l = [0.001, [trial.suggest_float("l_1", 0, 10), trial.suggest_float("l_2", 0, 10)]]

        N_n = len(cod_n)
        qubo = {}

        # 目的函数

        for s in range(N_n):
             add_qubo(qubo, ("x{}_{}".format(0, s), "x{}_{}".format(0, s)), l[0] * (dist(add_n, cod_n[s])))
             add_qubo(qubo, ("x{}_{}".format(N_n - 1, s), "x{}_{}".format(N_n - 1, s)), l[0] * (dist(add_n, cod_n[s])))

        for t in range(1, N_n - 1):
            for s in range(N_n):
                for s_ in range(N_n):
                    add_qubo(qubo, ("x{}_{}".format(t, s), "x{}_{}".format(t + 1, s_)), l[0] * dist(cod_n[s], cod_n[s_]))
        
        objective_qubo = qubo

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

        for s in range(N_n):
            if sum(x["x{}_{}".format(t, s)] for t in range(N_n)) != 1:
                Error_num += 1
        
        for t in range(N_n):
            if sum(x["x{}_{}".format(t, s)] for s in range(N_n)) != 1:
                Error_num += 1

        global value
        result_value = solver.sample_qubo(Q=objective_qubo, num_sweeps=0).first.energy
        
        if Error_num == 0 and result_value < value:

            value = result_value

            global turn_n
            for t in range(N_n):
                for s in range(N_n):
                    if x["x{}_{}".format(t, s)] == 1:
                        turn_n[n].append(Sn_n[s])

            # trial.study.stop()
        
        return l[1][0] + l[1][1] + 20 * Error_num

    return objective_n

for n in range(N):
    print("\n営業マン {}".format(n))

    Sn_n = list(S_n[n])
    add_n = add[n]
    cod_n = cod[Sn_n]

    value = np.inf
    study = optuna.create_study(direction="minimize")
    study.optimize(objective(n, Sn_n, add_n, cod_n), n_trials=200)


lines = []
color_line = []

for n in range(N):
    lines.append([tuple(add[n]), tuple(cod[turn_n[n][0]])])
    color_line.append(color[n])

    lines.append([tuple(add[n]), tuple(cod[turn_n[n][len(turn_n[n]) - 1]])])
    color_line.append(color[n])

    for s in range(len(turn_n[n]) - 1):
        lines.append([tuple(cod[turn_n[n][s]]), tuple(cod[turn_n[n][s + 1]])])
        color_line.append(color[n])

lc = mc.LineCollection(lines, colors=color_line, linewidth=2)

# 描画

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(aspect="1")
ax.add_collection(lc)
ax.autoscale()

plt.show()
