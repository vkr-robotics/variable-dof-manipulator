from controller import Supervisor, Keyboard
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
LAMBDA    = 0.1
STEP_SIZE = 0.8        # больше шаг — быстрее сходится
EPS       = 0.01

TOLERANCE = 0.005

# Кэш Якоби: пересчитываем раз в N кадров
JACOBIAN_REFRESH = 8

# Скорость мотора. Низкая = плавно. Высокая = быстро.
MOTOR_VELOCITY = 1.5

# Коэффициент сглаживания углов (0..1): 1 — без сглаживания
ALPHA = 0.35

# Шаг цели с клавиатуры
TARGET_STEP = 0.01

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

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

motors = {}
for name in JOINT_NAMES:
    motors[name] = robot.getDevice(name)
    if motors[name] is None:
        print(f"ОШИБКА: мотор '{name}' не найден"); exit(1)
    motors[name].setVelocity(MOTOR_VELOCITY)

sensors = {}
for name in SENSOR_NAMES:
    sensors[name] = robot.getDevice(name)
    if sensors[name] is None:
        print(f"ОШИБКА: сенсор '{name}' не найден"); exit(1)
    sensors[name].enable(timestep)

end_effector = robot.getFromDef(END_EFFECTOR_DEF)
if end_effector is None:
    print(f"ОШИБКА: не найдено DEF-имя '{END_EFFECTOR_DEF}'"); exit(1)


# ============================================================
#  Маркер цели
# ============================================================
def ensure_marker():
    node = robot.getFromDef("TARGET_MARKER")
    if node is None:
        root = robot.getRoot()
        children = root.getField("children")
        children.importMFNodeFromString(-1, f"""
        DEF TARGET_MARKER Solid {{
          translation {TARGET[0]} {TARGET[1]} {TARGET[2]}
          children [
            Shape {{
              geometry Sphere {{ radius 0.01 }}
              appearance PBRAppearance {{ baseColor 1 0 0 }}
            }}
          ]
        }}
        """)
    else:
        node.getField("translation").setSFVec3f(TARGET)

ensure_marker()


# ============================================================
#  Утилиты
# ============================================================
def step():
    if robot.step(timestep) == -1:
        exit(0)

def get_pos():
    return end_effector.getPosition()

def distance_to_target():
    p = get_pos()
    return math.sqrt(
        (p[0] - TARGET[0]) ** 2 +
        (p[1] - TARGET[1]) ** 2 +
        (p[2] - TARGET[2]) ** 2)

def get_joint_angles():
    return [sensors[n + "_sensor"].getValue() for n in JOINT_NAMES]

def set_joint_angles(angles):
    for i, name in enumerate(JOINT_NAMES):
        lo, hi = JOINT_LIMITS[name]
        motors[name].setPosition(max(lo, min(hi, angles[i])))


# ============================================================
#  Клавиатура
# ============================================================
def handle_keyboard():
    changed = False
    key = keyboard.getKey()
    while key != -1:
        ch = chr(key).lower() if 0 < key < 128 else ""
        if ch == "w":   TARGET[1] += TARGET_STEP; changed = True
        elif ch == "s": TARGET[1] -= TARGET_STEP; changed = True
        elif ch == "a": TARGET[0] -= TARGET_STEP; changed = True
        elif ch == "d": TARGET[0] += TARGET_STEP; changed = True
        elif ch == "q": TARGET[2] += TARGET_STEP; changed = True
        elif ch == "e": TARGET[2] -= TARGET_STEP; changed = True
        elif ch == "r":
            TARGET[0], TARGET[1], TARGET[2] = 0.25, 0.15, 0.35
            changed = True
        key = keyboard.getKey()

    if changed:
        ensure_marker()
    return changed


# ============================================================
#  Якоби (прямая разность, с опросом клавиатуры)
# ============================================================
def compute_jacobian_forward():
    """
    J[:, i] = (p(q_i + EPS) - p(q)) / EPS
    N замеров + N возвратов = 2N шагов.
    Внутри — опрос клавиатуры, чтобы не терять нажатия.
    """
    n = len(JOINT_NAMES)
    J = [[0.0] * n for _ in range(3)]

    base_angles = get_joint_angles()
    p_base = get_pos()

    for i, name in enumerate(JOINT_NAMES):
        # Опрос клавиатуры
        handle_keyboard()

        probe = base_angles[:]
        probe[i] += EPS
        set_joint_angles(probe)
        step()
        p_probe = get_pos()

        set_joint_angles(base_angles)
        step()

        for r in range(3):
            J[r][i] = (p_probe[r] - p_base[r]) / EPS

    return J


def solve_3x3(A, b):
    def det3(m):
        return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
              - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
              + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    d = det3(A)
    if abs(d) < 1e-12:
        return None
    out = []
    for i in range(3):
        M = [row[:] for row in A]
        for r in range(3):
            M[r][i] = b[r]
        out.append(det3(M) / d)
    return out


# ============================================================
#  DLS-шаг с готовой матрицей Якоби
# ============================================================
def apply_dls(J):
    p = get_pos()
    e = [TARGET[0] - p[0], TARGET[1] - p[1], TARGET[2] - p[2]]

    A = [[0.0]*3 for _ in range(3)]
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
        return

    dq = [sum(J[r][i] * y[r] for r in range(3))
          for i in range(len(JOINT_NAMES))]

    # Ограничение шага
    max_dq = max(abs(x) for x in dq) if dq else 0.0
    if max_dq > 0.2:
        k = 0.2 / max_dq
        dq = [x * k for x in dq]

    # Сглаживание: new = current + ALPHA * step
    current = get_joint_angles()
    new_angles = [current[i] + ALPHA * STEP_SIZE * dq[i]
                  for i in range(len(current))]
    set_joint_angles(new_angles)


# ============================================================
#  Основной цикл
# ============================================================
step()
print("=== Управление через ОЗК (DLS, оптимизированный) ===")
print("WASD/QE — движение цели, R — сброс.")

# Стартовая пауза, чтобы рука устоялась
for _ in range(10):
    step()

jacobian = None
jac_age = 0
frame = 0

while robot.step(timestep) != -1:
    frame += 1

    # 1) Клавиатура — каждый кадр
    handle_keyboard()

    # 2) Если цель далеко
    d = distance_to_target()
    if d > TOLERANCE:
        # Обновляем Якоби редко
        if jacobian is None or jac_age >= JACOBIAN_REFRESH:
            jacobian = compute_jacobian_forward()
            jac_age = 0
        jac_age += 1

        # Один маленький DLS-шаг
        apply_dls(jacobian)

    # 3) Редкий статус
    if frame % 60 == 0 and d > TOLERANCE:
        p = get_pos()
        print(f"d = {d*1000:.1f} мм, p = [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")