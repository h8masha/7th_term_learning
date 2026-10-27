import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

# ==========================================
# 1. ЦЕЛЕВАЯ ФУНКЦИЯ (Вариант 9)
# ==========================================
def f9(x):
    """Sum of different power function 9: sum(|x_i|^(i+1)), i=1:n"""
    x = np.asarray(x, dtype=float).flatten()
    n = len(x)
    powers = np.arange(2, n + 2)  # [2, 3, 4, ...]
    return np.sum(np.abs(x)**powers)

def f9_mesh(X, Y):
    return np.abs(X)**2 + np.abs(Y)**3

# ==========================================
# 2. КЛАСС НАШЕГО ГЕНЕТИЧЕСКОГО АЛГОРИТМА (RGA)
# ==========================================
class RGA:
    def __init__(self, n, pop_size=60, max_epochs=100, p_cross=0.9, p_mut=0.15,
                 lb=-1.0, ub=1.0, eta=2.0, tournament_k=3, seed=42):
        self.n = n
        self.pop_size = pop_size
        self.max_epochs = max_epochs
        self.p_cross = p_cross
        self.p_mut = p_mut
        self.lb = lb
        self.ub = ub
        self.eta = eta
        self.k = tournament_k
        self.rng = np.random.default_rng(seed)

        self.history = []
        self.best_x = None
        self.best_f = np.inf
        self.snapshots = []

    def init_population(self):
        return self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.n))

    def tournament(self, pop, fit):
        idx = self.rng.integers(0, len(pop), size=self.k)
        best = idx[np.argmin(fit[idx])]
        return pop[best].copy()

    def sbx(self, p1, p2):
        c1, c2 = p1.copy(), p2.copy()
        if self.rng.random() > self.p_cross:
            return c1, c2
        u = self.rng.random(self.n)
        beta = np.where(u <= 0.5,
                        (2*u)**(1.0/(self.eta+1)),
                        (1.0/(2*(1-u)))**(1.0/(self.eta+1)))
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return np.clip(c1, self.lb, self.ub), np.clip(c2, self.lb, self.ub)

    def mutate(self, ind, epoch):
        b = 2.0
        for i in range(self.n):
            if self.rng.random() < self.p_mut:
                r = self.rng.random()
                delta = 1 - r ** ((1 - epoch / self.max_epochs) ** b)
                if self.rng.random() < 0.5:
                    ind[i] = ind[i] + delta * (self.ub - ind[i])
                else:
                    ind[i] = ind[i] - delta * (ind[i] - self.lb)
        return np.clip(ind, self.lb, self.ub)

    def run(self, verbose=True):
        pop = self.init_population()
        fit = np.array([f9(ind) for ind in pop])

        for epoch in range(self.max_epochs):
            best_idx = np.argmin(fit)
            best_ind, best_val = pop[best_idx].copy(), fit[best_idx]

            if best_val < self.best_f:
                self.best_f = best_val
                self.best_x = best_ind.copy()

            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self.tournament(pop, fit)
                p2 = self.tournament(pop, fit)
                c1, c2 = self.sbx(p1, p2)
                c1 = self.mutate(c1, epoch)
                c2 = self.mutate(c2, epoch)
                offspring.extend([c1, c2])
            offspring = np.array(offspring[:self.pop_size])

            pop = np.vstack([offspring, best_ind])
            fit = np.array([f9(ind) for ind in pop])

            order = np.argsort(fit)
            pop = pop[order][:self.pop_size]
            fit = fit[order][:self.pop_size]

            self.history.append(fit[0])
            self.snapshots.append(pop.copy())

            if verbose and (epoch % 10 == 0 or epoch == self.max_epochs - 1):
                print(f"Эпоха {epoch:3d}: f_best = {fit[0]:.6e}")

        return self.best_x, self.best_f

# ==========================================
# 3. ЗАПУСК И ВИЗУАЛИЗАЦИЯ (n=2)
# ==========================================
print("=== n = 2 ===")
ga2 = RGA(n=2, pop_size=60, max_epochs=100, p_cross=0.9, p_mut=0.15)
best_x2, best_f2 = ga2.run(verbose=True)
print(f"Лучшее решение: x = {best_x2}, f = {best_f2:.6e}")

# График 3D для n=2 с кадрами эволюции
fig = plt.figure(figsize=(15, 5))
X = np.linspace(-1, 1, 50)
Y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(X, Y)
Z = f9_mesh(X, Y)

epochs_to_show = [0, 10, 50, 99]
for i, ep in enumerate(epochs_to_show):
    ax = fig.add_subplot(1, 4, i+1, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    pop = ga2.snapshots[ep]
    ax.scatter(pop[:, 0], pop[:, 1], [f9(p) for p in pop],
               color='red', s=10)
    ax.scatter([0], [0], [0], color='gold', marker='*', s=150, zorder=5)
    ax.set_title(f"Эпоха {ep+1}")
    ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('f(x)')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
plt.tight_layout()
plt.show()

# ==========================================
# 4. СРАВНЕНИЕ С БИБЛИОТЕКОЙ Pymoo
# ==========================================
print("\n=== Сравнение с pymoo GA (n=2) ===")

class F9Problem(Problem):
    def __init__(self, n):
        super().__init__(n_var=n, n_obj=1, n_constr=0, xl=-1.0, xu=1.0)
    def _evaluate(self, x, out, *args, **kwargs):
        powers = np.arange(2, self.n_var + 2)
        out["F"] = np.sum(np.abs(x)**powers, axis=1).reshape(-1, 1)

algorithm = GA(pop_size=60, eliminate_duplicates=True)
res = minimize(F9Problem(n=2), algorithm, ('n_gen', 100),
               verbose=False, seed=42)
x_best = np.array(res.X).flatten()
f_best = np.array(res.F).flatten()[0]
print(f"Pymoo GA, n=2: x = {x_best}, f = {f_best:.6e}")

# ==========================================
# 5. ЗАПУСК ДЛЯ n=3
# ==========================================
print("\n=== n = 3 ===")
ga3 = RGA(n=3, pop_size=60, max_epochs=100, p_cross=0.9, p_mut=0.15)
best_x3, best_f3 = ga3.run(verbose=True)
print(f"Лучшее решение: x = {best_x3}, f = {best_f3:.6e}")

print("\n=== Сравнение с pymoo GA (n=3) ===")
res3 = minimize(F9Problem(n=3), algorithm, ('n_gen', 100),
                verbose=False, seed=42)
x_best3 = np.array(res3.X).flatten()
f_best3 = np.array(res3.F).flatten()[0]
print(f"Pymoo GA, n=3: x = {x_best3}, f = {f_best3:.6e}")

# ==========================================
# 6. ЭКСПЕРИМЕНТЫ (Зависимость от N, Pc, Pm)
# ==========================================
print("\n--- Зависимость от N (n=2) ---")
print("  N  | Ошибка")
for N in [20, 50, 100, 200]:
    ga_tmp = RGA(n=2, pop_size=N, max_epochs=100, seed=42)
    _, best = ga_tmp.run(verbose=False)
    print(f"{N:3d} | {best:.2e}")

print("\n--- Зависимость от Pc (n=2, N=60) ---")
print(" Pc  | Ошибка")
for pc in [0.5, 0.7, 0.9, 1.0]:
    ga_tmp = RGA(n=2, pop_size=60, p_cross=pc, max_epochs=100, seed=42)
    _, best = ga_tmp.run(verbose=False)
    print(f"{pc:.1f} | {best:.2e}")

print("\n--- Зависимость от Pm (n=2, N=60, Pc=0.9) ---")
print(" Pm  | Ошибка")
for pm in [0.01, 0.05, 0.1, 0.15, 0.3]:
    ga_tmp = RGA(n=2, pop_size=60, p_cross=0.9, p_mut=pm,
                 max_epochs=100, seed=42)
    _, best = ga_tmp.run(verbose=False)
    print(f"{pm:.2f} | {best:.2e}")

print("\nГотово.")