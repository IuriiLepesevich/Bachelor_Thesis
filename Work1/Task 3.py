import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 10)
plt.ion()


def rotation_mat(degree):
    R = np.identity(3)
    Deg_Rad = np.radians(degree)
    c = np.cos(Deg_Rad)
    s = np.sin(Deg_Rad)
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    return R


def translation_mat(dx, dy):
    T = np.identity(3)
    T = np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])
    return T


def skew_mat(lx, ly):
    N = np.identity(3)
    N = np.array([
        [1, ly, 0],
        [lx, 1, 0],
        [0, 0, 1]
    ])
    return N


def scale_mat(sx, sy):
    S = np.identity(3)
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])
    return S


def dot(X, Y):
    try:
        if Y.ndim == 1 and X.ndim == 1:
            c = np.zeros(len(X))
            if X.ndim == 1:
                lenght = len(X)
            else:
                lenght = len(X[0])
            for k in range(len(X)):
                t = 0
                for i in range(lenght):
                    t += X[k] * Y[i]
                c[k] = t
        elif Y.ndim == 1 or X.ndim == 1:
            if X.ndim == 1:
                X, Y = Y, X
            c = np.zeros(len(X))
            if X.ndim == 1:
                lenght = len(X)
            else:
                lenght = len(X[0])
            for k in range(len(X)):
                t = 0
                for i in range(lenght):
                    t += X[k][i] * Y[i]
                c[k] = t
        else:
            c = np.zeros((len(X), len(Y[0])))
            for k in range(len(X)):
                for i in range(len(Y[0])):
                    t = 0
                    for j in range(len(Y)):
                        t += X[k][j] * Y[j][i]
                    c[k][i] = t
    except:
        print("Invalid input in dot() function")
        return
    Y = c
    return Y


def vec2d_to_vec3d(vec2):
    I = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])
    vec3 = dot(I, vec2) + np.array([0, 0, 1])
    return vec3


def vec3d_to_vec2d(vec3):
    I = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])
    vec2 = dot(I, vec3)
    return vec2


A = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 4]
])

B = np.array([
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]
])


class Character(object):
    def __init__(self):
        super().__init__()
        self.__angle = np.random.random() * np.pi

        self.geometry = []
        self.color = 'r'
        self.pos = np.array([0.0, 0.0])

        self.C = np.identity(3)
        self.R = rotation_mat(self.__angle)
        self.T = translation_mat(self.pos[0], self.pos[1])
        self.S = scale_mat(0.5, 1)

        self.speed = 0.1
        self.dir_init = np.array([0.0, 1.0])
        self.dir = np.array(self.dir_init)

        self.generate_geometry()

    def set_angle(self, angle):
        self.__angle = angle
        self.R = rotation_mat(self.__angle)

    def get_angle(self):
        return self.__angle

    def move(self):
        pass

    def draw(self):
        x_data = []
        y_data = []

        self.C = dot(self.T, self.R)
        self.C = dot(self.C, self.S)

        for vec2d in self.geometry:
            vec3d = vec2d_to_vec3d(vec2d)

            vec3d = dot(self.C, vec3d)

            vec2d = vec3d_to_vec2d(vec3d)
            x_data.append(vec2d[0])
            y_data.append(vec2d[1])

        plt.plot(x_data, y_data, self.color)

    def generate_geometry(self):
        pass

    def f_check(self):
        pass


class PLayer(Character):
    def __init__(self):
        super().__init__()
        self.pos = np.array([5.0, 0.0])

    def generate_geometry(self):
        self.geometry = np.array([
            [-1, 0],
            [1, 0],
            [0, 1],
            [-1, 0]
        ])

    def move(self):
        vec3d = vec2d_to_vec3d(self.dir)

        vec3d = dot(self.R, vec3d)

        vec2d = vec3d_to_vec2d(vec3d)
        self.pos[0] += vec2d[0] * self.speed
        self.pos[1] += vec2d[1] * self.speed

        self.T = translation_mat(self.pos[0], self.pos[1])

    def get_R(self):
        I = self.R
        return I

    def get_pos(self):
        I = self.pos
        return I

    # Here is wrong implementation of task 7, as I had problems with getting radius variable from Asteroid() class.
    # It still works, but you won't last more than a few second in game, as collision is too big
    '''def f_check(self):
        for entity in characters:
            if isinstance(entity, Asteroid):
                if abs(self.pos[0] - entity.pos[0]) <= 0.005 or abs(
                        self.pos[0] - entity.pos[1]) <= 0.005 or abs(
                    self.pos[1] - entity.pos[1]) <= 0.005 or abs(
                    self.pos[1] - entity.pos[0]) <= 0.005:
                    exit()'''


class Asteroid(Character):
    def __init__(self):
        super().__init__()
        self.pos = np.array([np.random.uniform(-9, 10), np.random.uniform(-9, 10)])
        self.T = translation_mat(self.pos[0], self.pos[1])
        self.color = (np.random.random(), np.random.random(), np.random.random())
        self.S = skew_mat(np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3))
        self.speed = np.random.uniform(0.4, 1.5)
        self.dir_init = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)])
        self.dir = np.array(self.dir_init)

    def generate_geometry(self):
        theta = 0
        radius = (np.random.uniform(0.3, 1))
        while theta <= 360:
            if theta % 20 == 0:
                self.geometry.append([self.pos[0] + (radius * np.cos(np.radians(theta))) + np.random.uniform(0.01, 0.1),
                                      self.pos[1] + (radius * np.sin(np.radians(theta))) + np.random.uniform(0.01,
                                                                                                             0.1)])
            theta += 1
        self.geometry = np.array(self.geometry)

    def draw(self):
        x_data = []
        y_data = []

        self.C = dot(self.S, self.T)
        for vec2d in self.geometry:
            vec3d = vec2d_to_vec3d(vec2d)

            vec3d = dot(self.C, vec3d)

            vec2d = vec3d_to_vec2d(vec3d)

            x_data.append(vec2d[0])
            y_data.append(vec2d[1])

        plt.plot(x_data, y_data, c=self.color)

    def move(self):
        if self.pos[0] <= -10 or self.pos[0] >= 10:
            self.dir[0] = -self.dir[0]

        if self.pos[1] <= -10 or self.pos[1] >= 10:
            self.dir[1] = -self.dir[1]

        vec3d = vec2d_to_vec3d(self.dir)

        vec2d = vec3d_to_vec2d(vec3d)
        self.pos[0] += vec2d[0] * self.speed
        self.pos[1] += vec2d[1] * self.speed

        self.T = translation_mat(self.pos[0], self.pos[1])


class Rocket(Character):
    def __init__(self):
        super().__init__()
        self.S = scale_mat(0.1, 0.2)
        self.speed = 1
        self.pos = np.array(player.get_pos())
        self.R = player.get_R()
        self.dir = np.array(player.dir)

    def generate_geometry(self):
        self.geometry = np.array([
            [-1, 0],
            [1, 0],
            [0, 1],
            [-1, 0]
        ])

    def draw(self):
        x_data = []
        y_data = []

        vec3d = vec2d_to_vec3d(self.dir)

        vec3d = dot(self.R, vec3d)

        vec2d = vec3d_to_vec2d(vec3d)
        self.pos[0] += vec2d[0] * self.speed
        self.pos[1] += vec2d[1] * self.speed

        self.T = translation_mat(self.pos[0], self.pos[1])

        self.C = dot(self.T, self.R)
        self.C = dot(self.C, self.S)

        for vec2d in self.geometry:
            vec3d = vec2d_to_vec3d(vec2d)

            vec3d = dot(self.C, vec3d)

            vec2d = vec3d_to_vec2d(vec3d)
            x_data.append(vec2d[0])
            y_data.append(vec2d[1])

        plt.plot(x_data, y_data, 'b')


characters = []
characters.append(PLayer())
for i in range(10):
    characters.append(Asteroid())
player = characters[0]

is_running = True


def on_press(event):
    global is_running, player
    if event.key == 'escape':
        is_running = False
    elif event.key == 'left':
        player.set_angle(player.get_angle() + 5)
    elif event.key == 'right':
        player.set_angle(player.get_angle() - 5)
    elif event.key == ' ':
        characters.append(Rocket())


fig, _ = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)

while is_running:
    plt.clf()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    for each in characters:
        each.f_check()
        each.draw()
        each.move()
        plt.title(f"angle: {player.get_angle() % 360}")

    plt.draw()
    plt.pause(1e-2)
