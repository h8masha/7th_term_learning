import sys
import time
import pandas as pd

# Защита от переполнения стека (как в отчете одногруппника)
sys.setrecursionlimit(100000)

# ---------------- Итеративная реализация ----------------
def f_iter(x, y):
    """
    Итеративная реализация f(x, y) = x * y + 2x.
    Считаем через циклы, имитируя суперпозицию add(mult(x, y), dbl(x)).
    """
    if x < 0 or y < 0:
        raise ValueError("Аргументы должны быть неотрицательными")
    
    # Часть 1: Вычисляем mult(x, y) = x * y
    product = 0
    for _ in range(x):
        product += y
        
    # Часть 2: Вычисляем dbl(x) = 2 * x
    double_x = 0
    for _ in range(x):
        double_x += 2
        
    # Часть 3: Сложение (add)
    return product + double_x

# ---------------- Рекурсивная реализация (по формальной схеме) ----------------
def add_rec(a, b):
    """Сложение: add(a, b) = a + b"""
    if a == 0:
        return b
    return 1 + add_rec(a - 1, b)

def mult_rec(a, b):
    """Умножение: mult(a, b) = a * b"""
    if a == 0:
        return 0
    return add_rec(mult_rec(a - 1, b), b)

def dbl_rec(n):
    """Удвоение: dbl(n) = 2 * n"""
    if n == 0:
        return 0
    return 2 + dbl_rec(n - 1) # Эквивалентно S(S(dbl(n-1)))

def f_rec(x, y):
    """Целевая функция: f(x, y) = x * y + 2x (через суперпозицию)"""
    return add_rec(mult_rec(x, y), dbl_rec(x))

# ---------------- Сравнение ----------------
def benchmark(pairs):
    results = []
    for x, y in pairs:
        t0 = time.perf_counter()
        r_iter = f_iter(x, y)
        t1 = time.perf_counter()
        
        r_rec = f_rec(x, y)
        t2 = time.perf_counter()
        
        assert r_iter == r_rec, f"Несовпадение при x={x}, y={y}: {r_iter} != {r_rec}"
        
        results.append({
            "x": x,
            "y": y,
            "f(x,y)": r_iter,
            "Итеративно (мс)": (t1 - t0) * 1000,
            "Рекурсивно (мс)": (t2 - t1) * 1000,
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Тестовые наборы (такие же, как у одногруппника, чтобы наглядно сравнить)
    test_pairs = [(5, 3), (10, 7), (20, 15), (50, 40), (100, 80), (200, 150)]
    df = benchmark(test_pairs)
    
    # Вывод красивой таблицы
    print(df.to_string(index=False))