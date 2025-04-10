import numpy as np
from numpy.linalg import norm, inv, cond
from gauss import gauss
from householder import householder
from progon import progon

def run_all_methods(A, b, x_exact, label):
    print(f"\n===== {label} =====")
    
    for method_name, solver in [("Метод Гаусса", gauss), ("Метод Хаусхолдера", householder)]:
        x = solver(A.copy(), b.copy())

        r = A @ x - b
        error = x - x_exact

        print(f"\n{method_name}")
        print("x =", x)
        print("1-норма невязки:", norm(r, 1))
        print("∞-норма невязки:", norm(r, np.inf))
        print("1-норма погрешности:", norm(error, 1))
        print("∞-норма погрешности:", norm(error, np.inf))

        relative_error_1 = norm(error, 1) / norm(x_exact, 1)
        relative_error_inf = norm(error, np.inf) / norm(x_exact, np.inf)
        print("Относительная 1-норма погрешности:", relative_error_1)
        print("Относительная ∞-норма погрешности:", relative_error_inf)

    # Обратная матрица и проверка
    A_inv = inv(A)
    I_approx = A_inv @ A
    print("\nОбратная матрица A⁻¹:")
    print(A_inv)
    print(f"Норма (E - A⁻¹A): {norm(np.eye(len(A)) - A_inv @ A):.3e}")
    
    # Обусловленность
    cond_1 = cond(A, 1)
    cond_inf = cond(A, np.inf)
    print(f"cond_1(A): {cond_1:.3e}")
    print(f"cond_inf(A): {cond_inf:.3e}")


    if cond_1 < 100:
        print("➤ Матрица хорошо обусловлена.")
    else:
        print("➤ Матрица плохо обусловлена. Результаты могут быть неточными.")

# ============ Вариант 2 ============

# 1. Хорошо обусловленная система
A1 = np.array([
    [-52.4, 0.0, -0.57, 4.73],
    [0.12, 32.4, 9.05, 0.49],   
    [0.0, 5.88, -175.0, 2.43],   
    [-5.01, -2.43, 1.87, -76.2]
])    

b1 = np.array([-1309.17, 224.13, 97.4, -7800.62])
x1_exact = np.array([34, 5, 1, 100])
run_all_methods(A1, b1, x1_exact, "Система 1: Хорошо обусловленная")

# 2. Плохо обусловленная система
A2 = np.array([
    [-558.5500, 2569.1600, 12.6000, 55.8600],
    [-139.6500, 642.3400, 3.1500, 13.9650],   
    [-19.9500, 91.7700, 0.4000, 1.9950],
    [838.3900, -3855.9940, -18.9000, -83.7890] 
])

b2 = np.array([127.31, 31.815, 2.045, -190.399])
x2_exact = np.array([1, 0, 50, 1])

run_all_methods(A2, b2, x2_exact, "Система 2: Плохо обусловленная")
# -----
print()
print()
# Данные варианта 2
a = np.array([1 , 1, -1 , 2 ,-1], dtype=float)  # длина = n-1
c = np.array([60 ,  80,  130 , -90 , 140 ,  70], dtype=float)  # длина = n
b = np.array([1 , 1 ,-2 , 1, -1], dtype=float)  # длина = n-1
d = np.array([6  , 7 , 13,  -8 , 15,   9], dtype=float)  # длина = n
# Решение методом прогонки
x = progon(a, c, b, d)
# Собираем A для проверки (только для проверки!)
n = len(c)
A = np.zeros((n, n))
for i in range(n):
    A[i, i] = c[i]
    if i > 0:
        A[i, i - 1] = a[i - 1]
    if i < n - 1:
        A[i, i + 1] = b[i]

# Вычисление невязки
r = A @ x - d
print("Решение методом прогонки:")
print("x =", x)
print("1-норма невязки:", np.linalg.norm(r, 1))
print("∞-норма невязки:", np.linalg.norm(r, np.inf))
