from controller import Supervisor
import math

# ============================================================
#  Настройки
# ============================================================
TIME_STEP = 32

JOINT_NAMES  = ["joint_1", "joint_2", "joint_3", "joint_4"]
SENSOR_NAMES = [n + "_sensor" for n in JOINT_NAMES]

END_EFFECTOR_DEF = "tip"
TARGET = [0.25, 0.15, 0.35]

# Параметры DLS
MAX_ITER  = 200
TOLERANCE = 0.005
LAMBDA    = 0.1        # увеличил регуляризацию
STEP_SIZE = 0.3        # чуть мягче
EPS       = 0.02       # шаг для численного дифференцирования

# Скорость суставов
MOTOR_VELOCITY = 2.0

# Сколько шагов симуляции ждать после команды,
# чтобы сустав реально дошёл и сенсоры «успокоились»
SETTLE_STEPS = 3

# Лимиты: continuous-суставы не должны обрезаться
JOINT_LIMITS = {
    "joint_1": (-100.0, 100.0),
    "joint_2": (-100.0, 100.0),
    "joint_3": (-100.0, 100.0),
    "joint_4": (-100.0, 100.0),
}

# ============================================================
#  Инициализация
# ============================================================
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

motors = {}
for name in JOINT_NAMES:
    motors[name] = robot.getDevice(name)
    if motors[name] is None:
        print(f"ОШИБКА: мотор '{name}' не найден")
        exit(1)
    motors[name].setVelocity(MOTOR_VELOCITY)

sensors = {}
for name in SENSOR_NAMES:
    sensors[name] = robot.getDevice(name)
    if sensors[name] is None:
        print(f"ОШИБКА: сенсор '{name}' не найден")
        exit(1)
    sensors[name].enable(timestep)

end_effector = robot.getFromDef(END_EFFECTOR_DEF)
if end_effector is None:
    print(f"ОШИБКА: не найдено DEF-имя '{END_EFFECTOR_DEF}'")
    exit(1)


# ============================================================
#  Утилиты
# ============================================================
def step():
    if robot.step(timestep) == -1:
        exit(0)


def settle(n=SETTLE_STEPS):
    for _ in range(n):
        step()


def get_pos():
    return end_effector.getPosition()


def distance_to_target():
    p = get_pos()
    return math.sqrt(
        (p[0] - TARGET[0]) ** 2 +
        (p[1] - TARGET[1]) ** 2 +
        (p[2] - TARGET[2]) ** 2
    )


def get_joint_angles():
    return [sensors[n + "_sensor"].getValue() for n in JOINT_NAMES]


def set_joint_angles(angles):
    for i, name in enumerate(JOINT_NAMES):
        lo, hi = JOINT_LIMITS[name]
        a = max(lo, min(hi, angles[i]))
        motors[name].setPosition(a)


def compute_jacobian():
    """Численная матрица Якоби 3×N."""
    n = len(JOINT_NAMES)
    J = [[0.0] * n for _ in range(3)]

    base_angles = get_joint_angles()

    for i, name in enumerate(JOINT_NAMES):
        # +EPS
        angles_plus = base_angles[:]
        angles_plus[i] += EPS
        set_joint_angles(angles_plus)
        settle()
        p_plus = get_pos()

        # -EPS
        angles_minus = base_angles[:]
        angles_minus[i] -= EPS
        set_joint_angles(angles_minus)
        settle()
        p_minus = get_pos()

        # Возврат к базовым
        set_joint_angles(base_angles)
        settle()

        for r in range(3):
            J[r][i] = (p_plus[r] - p_minus[r]) / (2.0 * EPS)

    return J


# ============================================================
#  DLS
# ============================================================
def solve_dls():
    print(f"Цель: {TARGET}")
    print(f"Стартовые углы: {[round(a, 3) for a in get_joint_angles()]}")

    # Дать руке успокоиться перед первым расчётом Якоби
    settle(10)

    prev_d = distance_to_target()
    stall_count = 0

    for iteration in range(MAX_ITER):
        d = distance_to_target()

        if iteration % 5 == 0:
            p = get_pos()
            q = get_joint_angles()
            print(f"Итерация {iteration:3d}: "
                  f"d = {d:.4f} м, "
                  f"p = [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}], "
                  f"q = [{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}]")

        if d < TOLERANCE:
            print(f"Цель достигнута за {iteration} итераций.")
            return True

        # Диагностика: показать Якоби на первой итерации
        if iteration == 0:
            J = compute_jacobian()
            print("Матрица Якоби (3×N) на первой итерации:")
            for r in range(3):
                print("  " + "  ".join(f"{J[r][c]:+.4f}" for c in range(len(JOINT_NAMES))))
        else:
            J = compute_jacobian()

        # Вектор ошибки
        p = get_pos()
        e = [TARGET[0] - p[0], TARGET[1] - p[1], TARGET[2] - p[2]]

        # A = J J^T + λ² I
        A = [[0.0] * 3 for _ in range(3)]
        for r in range(3):
            for c in range(3):
                s = 0.0
                for k in range(len(JOINT_NAMES)):
                    s += J[r][k] * J[c][k]
                A[r][c] = s
        for r in range(3):
            A[r][r] += LAMBDA ** 2

        y = solve_3x3(A, e)
        if y is None:
            print("Матрица вырождена, пропуск итерации.")
            continue

        dq = [0.0] * len(JOINT_NAMES)
        for i in range(len(JOINT_NAMES)):
            s = 0.0
            for r in range(3):
                s += J[r][i] * y[r]
            dq[i] = s

        # Клип на шаг
        max_dq = max(abs(x) for x in dq) if dq else 0.0
        if max_dq > 0.15:
            scale = 0.15 / max_dq
            dq = [x * scale for x in dq]

        current = get_joint_angles()
        new_angles = [current[i] + STEP_SIZE * dq[i] for i in range(len(current))]
        set_joint_angles(new_angles)
        settle()

        # Диагностика застоя
        if abs(d - prev_d) < 1e-5:
            stall_count += 1
            if stall_count >= 3:
                print("Прогресс остановился — дальше не сойдётся.")
                return False
        else:
            stall_count = 0
        prev_d = d

    print("Не сошлось за отведённые итерации.")
    return False


def solve_3x3(A, b):
    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
              - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
              + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    d = det3(A)
    if abs(d) < 1e-12:
        return None

    result = []
    for i in range(3):
        M = [row[:] for row in A]
        for r in range(3):
            M[r][i] = b[r]
        result.append(det3(M) / d)
    return result


# ============================================================
#  Запуск
# ============================================================
step()
print("Контроллер запущен. Решение ОЗК методом DLS.")

success = solve_dls()

if success:
    print("Удержание позиции. Ctrl+C для выхода.")
else:
    print("Цель не достигнута.")

while robot.step(timestep) != -1:
    pass