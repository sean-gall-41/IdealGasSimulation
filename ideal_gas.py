import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# GLOBAL TODO: Ensure that changing certain parameters (particle radius, box dimensions, etc.)
# is robust ie that changes to these parameters do not change eg how close it looks like balls 
# get to walls. This requires some parameters to be proportional to each other (eg particle radius
# with marker size)

#GLOBAL VARIABLES

X_LOWER, X_UPPER, Y_LOWER, Y_UPPER = -6.0, 6.0, -6.0, 6.0
PARTICLE_MASS, PARTICLE_RADIUS = 0.01, 0.035
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

    # predicate function to determine whether the caller collides with
    # the parameter particle
    def collides_with(self, particle):
        D =  np.sqrt((self.position[0] - particle.position[0])**2 + 
                     (self.position[1] - particle.position[1])**2)
        
        return True if D < self.radius + particle.radius else False


# A particle box represents a group of N gas particles confined
# within a box of dimensions bounds
class ParticleBox():
    def __init__(self, N=25, bounds=[-1.0, 1.0, -1.0, 1.0]):
        self.__t = 0.0
        self.__N = N
        self.__bounds = bounds
        self.__particle_list = np.array([GasParticle(PARTICLE_MASS, PARTICLE_RADIUS,
                                        [-0.5*BOX_WIDTH + BOX_WIDTH*np.random.random(), 
                                        -0.5*BOX_HEIGHT + BOX_HEIGHT*np.random.random()],
                                        [-1.0 + np.random.random(), -1.0 + np.random.random()]) 
                                        for i in np.arange(N)])

    @property
    def t(self):
        return self.__t
    
    @t.setter
    def t(self, t):
        self.__t = t
    
    @property
    def N(self):
        return self.__N

    @N.setter
    def N(self, N):
        self.__N = N

    @property 
    def bounds(self):
        return self.__bounds

    @bounds.setter
    def bounds(self, bounds):
        self.__bounds = bounds
    
    @property
    def particle_list(self):
        return self.__particle_list

    # TODO: should probably run some kind of input validation on this...
    @particle_list.setter
    def particle_list(self, list):
        self.particle_list = list

    # TODO: migrate step function from GasParticle class over here. Then
    # have step return the updated list of positions and velocities.
    # This also means you should combine position and velocity lists into 
    # a single list, for easier readability and more maintainable code (like
    # when you access positions in animate function: just use list-splicing)
    # Steps the state of the particle box forward by time dt
    # => all particle positions and velocities are updated. 
    
    #naively: just run a for loop lol
    def step(self, dt):
        self.t += dt
        # print('%.1fs' % self.t)
        # TODO: add particle interactions
        for particle in self.particle_list:
            if particle.position[0] < particle.radius + self.bounds[0]:
                particle.position[0] = particle.radius + self.bounds[0]
                particle.velocity[0] *= -1
            if particle.position[0] > -particle.radius + self.bounds[1]:
                particle.position[0] = -particle.radius + self.bounds[1]
                particle.velocity[0] *= -1
            if particle.position[1] < particle.radius + self.bounds[2]:
                particle.position[1] = particle.radius + self.bounds[2]
                particle.velocity[1] *= -1
            if particle.position[1] > -particle.radius + self.bounds[3]:
                particle.position[1] = -particle.radius + self.bounds[3]
                particle.velocity[1] *= -1

            particle.position += dt*particle.velocity


# Set up figure
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                      xlim=(X_LOWER, X_UPPER), ylim=(Y_LOWER, Y_UPPER))

ax.set_title('Ideal Gas In a Box')
time_display = ax.text(0.25, 0.2, '', transform=ax.transAxes)

offset = 0.05 # an offset to make particles look like they are actually bouncing off walls
# TODO: link marker size to box collisions and ensure that regardless which parameter you change,
#       the wall collision behaviour is unchanged (ie balls still look like they hit walls)
rect = plt.Rectangle((-0.5*BOX_WIDTH-offset, -0.5*BOX_HEIGHT-offset), BOX_WIDTH+2*offset, 
                     BOX_HEIGHT+2*offset, lw=1.5, ec='k', fc='none')

ax.add_patch(rect)

np.random.seed(0)
my_box = ParticleBox(100, [-0.5*BOX_WIDTH, 0.5*BOX_WIDTH, -0.5*BOX_HEIGHT, 0.5*BOX_HEIGHT])

#TODO: find correct conversion factor for radius to marker size (in points)
marker_size = int(fig.dpi*2*PARTICLE_RADIUS*fig.get_figwidth()
                  / np.diff(ax.get_xbound())[0])

particle_pos, = ax.plot([], [], 'bo', ms=marker_size)


dt = 1./30

def init():
    particle_pos.set_data([], [])
    time_display.set_text('')
    return particle_pos, time_display

def animate(i):
    global my_box, dt
    my_box.step(dt)
    
    particle_pos_x = [particle.position[0] for particle in my_box.particle_list]
    particle_pos_y = [particle.position[1] for particle in my_box.particle_list]
    
    particle_pos.set_data(particle_pos_x, particle_pos_y)
    time_display.set_text('time = %.1fs' % my_box.t)
    # for particle in my_box.particle_list:
    #     particle.step(dt, my_box.bounds)
    #     particle_pos_x.append(particle.position[0])
    #     particle_pos_y.append(particle.position[1])
    return particle_pos, time_display

anim = animation.FuncAnimation(fig, animate, frames=30, interval=10, blit=True,
                               init_func=init)

plt.show()        



