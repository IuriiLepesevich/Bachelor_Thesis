import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 10)
plt.ion()


### Part 1


def func1(param1, param2):
    if param2 == 0:
        return 1
    elif param2 > 0:
        return param1 * func1(param1, param2 - 1)
    else:
        return func1(param1, param2 + 1) / param1


print(func1(param1=-2, param2=-3))


### Part 2


class Animal(object):
    def __init__(self):
        super().__init__()
        self._hunger_perc = 0.5

    def get_hunger_perc(self):
        return self._hunger_perc

    def eat(self):
        self._hunger_perc -= 0.1
        self._hunger_perc = max(0, self._hunger_perc)

    def sleep(self, hours):
        self._hunger_perc += hours * 0.1
        self._hunger_perc = min(1, self._hunger_perc)

    def move(self):
        pass


class Dog(Animal):
    def __init__(self):
        super().__init__()
        self.__bones_hidden = 0

    def move(self):
        self._hunger_perc += 0.1
        self._hunger_perc = min(1, self._hunger_perc)

    def bark(self):
        print('Bark')


class Cat(Animal):
    def __init__(self):
        super().__init__()
        self.__items_destroyed = 0

    def move(self):
        self._hunger_perc += 1e-2
        self._hunger_perc = min(1, self._hunger_perc)

    def meow(self):
        print('Meow')


class Robot(object):
    def __init__(self):
        super().__init__()
        self.__battery_perc = 1.0

    def move(self):
        self.__battery_perc -= 1e-1
        self.__battery_perc = max(0, self.__battery_perc)

    def charge(self, hours):
        self.__battery_perc += 1e-1 * hours
        self.__battery_perc = min(1, self.__battery_perc)


who_is_in_the_room = []
who_is_in_the_room.append(Dog())
dog_b = Dog()
who_is_in_the_room.append(dog_b)
who_is_in_the_room.append(Cat())
who_is_in_the_room.append(Robot())

for entity in who_is_in_the_room:
    entity.move()
    if isinstance(entity, Animal):
        print(f'Hunger = {entity.get_hunger_perc()}')
        entity.eat()
        if isinstance(entity, Dog):
            entity.bark()
    else:
        if isinstance(entity, Robot):
            entity.charge(hours=2)


### Part 3


class Character(object):
    def __init__(self):
        super().__init__()
        self.geometry = []
        self.angle = 0.0
        self.speed = 0.1
        self.pos = np.array([0, 0])
        self.dir = np.array([0, 1])
        self.color = 'r'
        self.C = np.identity(3)
        self.R = np.identity(3)
        self.T = np.identity(3)

        self.generate_geometry()

    def draw(self):
        x_data = []
        y_data = []
        for vec2 in self.geometry:
            x_data.append(vec2[0])
            y_data.append(vec2[1])
        plt.plot(x_data, y_data, self.color)

    def generate_geometry(self):
        pass


class PLayer(Character):
    def __init__(self):
        super().__init__()

    def generate_geometry(self):
        self.length = 1
        self.geometry.append([self.pos[0] - self.length/2, self.pos[1] - self.length/2])
        self.geometry.append([self.pos[0], self.pos[1] + self.length/2])
        self.geometry.append([self.pos[0] + self.length/2, self.pos[1] - self.length/2])
        self.geometry.append(self.geometry[0])


class Asteroid(Character):
    def __init__(self):
        super().__init__()

    def generate_geometry(self):
        self.geometry.append([self.pos, self.dir])


characters = []
player = PLayer()
characters.append(player)

characters.append(Asteroid())
characters.append(Asteroid())

is_running = True


def on_press(event):
    global is_running, player
    if event.key == 'escape':
        is_running = False
    elif event.key == 'left':
        player.angle += 5
    elif event.key == 'right':
        player.angle -= 5

fig, _ = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)

while is_running:
    plt.clf()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    for each in characters:
        each.draw()
        plt.title(f"angle: {player.angle}")

    plt.draw()
    plt.pause(1e-3)