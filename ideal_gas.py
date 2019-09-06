import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

#GLOBAL VARIABLES

X_LOWER, X_UPPER, Y_LOWER, Y_UPPER = -5.0, 5.0, -5.0, 5.0
PARTICLE_MASS, PARTICLE_RADIUS = 0.01, 0.07
SCALE_FACTOR = 0.5
BOX_WIDTH = SCALE_FACTOR*(X_UPPER-X_LOWER)
BOX_HEIGHT = SCALE_FACTOR*(Y_UPPER-Y_LOWER)

class GasParticle:
    """docstring for GasParticle."""

    """To understand how we are going to initialize our ball object, 
    we need to understand how to plot artists in matplotlib"""
    def __init__(self, mass = PARTICLE_MASS, radius = PARTICLE_RADIUS,
                 position = [0.0, 0.0], velocity = [0.0, 0.0]):
        self.__mass = mass
        self.__radius = radius
        self.__position = np.asarray(position, dtype=float)
        self.__velocity = np.asarray(velocity, dtype=float)

    @property
    def mass(self):
        return self.__mass

    @mass.setter
    def mass(self, mass):
        self.__mass = mass
    
    @property
    def radius(self):
        return self.__radius

    @radius.setter
    def radius(self, radius):
        self.__radius = radius

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, position):
        self.__position = position

    @property
    def velocity(self):
        return self.__velocity

    @velocity.setter
    def velocity(self, velocity):
        self.__velocity = velocity

    # bounds is a list rep the bounding box: [x_min, x_max, y_min, y_max]
    # TODO: fix wacky bouncing bug for off values/combinations of scale factor and radius
    #       Maybe particle is getting trapped within the distance calculated between frames?
    def step(self, dt, bounds):
        # TODO: Should probably make delta dependent on radius so that changing global var doesnt 
        #       mess everything up. A fair assessment: if we increase radius, we get a greater "window"
        #       for our particle to become trapped in. 

        if self.position[0] < self.radius + bounds[0]:
            self.position[0] = self.radius + bounds[0]
            self.velocity[0] *= -1
        if self.position[0] > -self.radius + bounds[1]:
            self.position[0] = -self.radius + bounds[1]
            self.velocity[0] *= -1
        if self.position[1] < self.radius + bounds[2]:
            self.position[1] = self.radius + bounds[2]
            self.velocity[1] *= -1
        if self.position[1] > -self.radius + bounds[3]:
            self.position[1] = -self.radius + bounds[3]
            self.velocity[1] *= -1


        # delta = 1.0E-2
        # epsilon = self.radius + delta
        # if abs(bounds[1]-self.position[0]) <= epsilon:
        #     self.velocity[0] = -self.velocity[0]
        # elif abs(bounds[0]-self.position[0]) <= epsilon:
        #     self.velocity[0] = -self.velocity[0]
        # elif abs(bounds[3]-self.position[1]) <= epsilon:
        #     self.velocity[1] = -self.velocity[1]
        # elif abs(bounds[2]-self.position[1]) <= epsilon:
        #     self.velocity[1] = -self.velocity[1]
        #now update positions regardless if collision occurred 
        self.position += dt*self.velocity

# Set up figure
fig = plt.figure()
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1) #Kind of unsure what this does?
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                      xlim=(X_LOWER, X_UPPER), ylim=(Y_LOWER, Y_UPPER))

ax.set_title('Ideal Gas In a Box')

rect = plt.Rectangle((-0.5*BOX_WIDTH, -0.5*BOX_HEIGHT), BOX_WIDTH, 
                     BOX_HEIGHT, lw=1.5, ec='k', fc='none')

ax.add_patch(rect)

#TODO: make sure particle trails fade away over time
my_particle = GasParticle(PARTICLE_MASS, PARTICLE_RADIUS,
                          [-0.5*BOX_WIDTH, 0.0],
                          [5.0, 2.5])

#TODO: find correct conversion factor for radius to marker size (in points)
marker_size = int(fig.dpi*2*my_particle.radius*fig.get_figwidth()
                  / np.diff(ax.get_xbound())[0])
particle_pos, = ax.plot([], [], 'bo', ms=marker_size)
pos_x = []
pos_y = []
particle_path, = ax.plot([], [], c='0.75', lw=1)

dt = 1./30

def init():
    particle_pos.set_data([], [])
    particle_path.set_data(pos_x, pos_y)
    return particle_pos, particle_path,

def animate(i):
    my_particle.step(dt, [-0.5*BOX_WIDTH, 0.5*BOX_WIDTH, -0.5*BOX_HEIGHT, 0.5*BOX_HEIGHT])
    pos_x.append(my_particle.position[0])
    pos_y.append(my_particle.position[1])
    particle_pos.set_data([my_particle.position[0]], [my_particle.position[1]])
    particle_path.set_data(pos_x, pos_y)
    return particle_pos, particle_path,

anim = animation.FuncAnimation(fig, animate, frames=30, interval=10, blit=True,
                               init_func=init)

plt.show()        



