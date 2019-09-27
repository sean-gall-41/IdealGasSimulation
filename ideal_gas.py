from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# GLOBAL TODO: Ensure that changing certain parameters (particle radius, box dimensions, etc.)
# is robust ie that changes to these parameters do not change eg how close it looks like balls 
# get to walls. This requires some parameters to be proportional to each other (eg particle radius
# with marker size)

#GLOBAL VARIABLES

X_LOWER, X_UPPER, Y_LOWER, Y_UPPER = -6.0, 6.0, -6.0, 6.0
PARTICLE_MASS, PARTICLE_RADIUS = 0.01, 0.04
SCALE_FACTOR = 0.75
BOX_WIDTH = SCALE_FACTOR*(X_UPPER-X_LOWER)
BOX_HEIGHT = SCALE_FACTOR*(Y_UPPER-Y_LOWER)

def randomReal(a, b):
    return (b - a) * np.random.random() + a

class Error(Exception):
    """Base class for other custom-defined exceptions (if the need arises)"""
    pass

class OutOfBoundsError(Error):
    """Raised when something is out of bounds"""
    pass

class GasParticle:
    """docstring for GasParticle."""

    """To understand how we are going to initialize our ball object, 
    we need to understand how to plot artists in matplotlib"""
    def __init__(self, mass = PARTICLE_MASS, radius = PARTICLE_RADIUS,
                 color = 'b', init_state = [0.0, 0.0, 0.0, 0.0]):
        self.mass = mass
        self.radius = radius
        self.init_state = np.asarray(init_state, dtype=float)
        self.state = self.init_state.copy()
        self.color = color

    def __eq__(self, other):
        return True if self.mass == other.mass and self.radius == other.radius and np.array_equal(self.state, other.state) else False

    # predicate function to determine whether the caller collides with
    # the parameter particle
    def collides_with(self, other):
        D =  np.sqrt((self.state[0] - other.state[0])**2 + 
                     (self.state[1] - other.state[1])**2)
        
        return True if D <= self.radius + other.radius else False


# A particle box represents a group of N gas particles confined
# within a box of dimensions bounds
class ParticleBox():
    def __init__(self, N=25, color='b', bounds=[-1.0, 1.0, -1.0, 1.0]):
        self.t = 0.0
        self.N = N
        self.color = color # Should the box itsef even have this attribute?
        self.bounds = bounds
        self.particle_list = np.array([GasParticle(PARTICLE_MASS, PARTICLE_RADIUS, color,
                                        [randomReal(-0.5*BOX_WIDTH, 0.5*BOX_WIDTH), 
                                        randomReal(-0.5*BOX_HEIGHT, 0.5*BOX_HEIGHT),
                                        randomReal(-5, 5), randomReal(-5, 5)]) 
                                        for i in np.arange(N)])

    def add_particle(self, particle):
        if particle.state[0] - particle.radius < self.bounds[0] or \
           particle.state[0] + particle.radius > self.bounds[1] or \
           particle.state[1] - particle.radius < self.bounds[2] or \
           particle.state[1] + particle.radius > self.bounds[3]:
            raise OutOfBoundsError 
        
        self.N += 1
        self.particle_list.append(particle)

    # Steps the state of the particle box forward by time dt
    # => all particle positions and velocities are updated. 
    def step(self, dt):
        self.t += dt
        
        # Somehow very cryptic code?
        D = squareform(pdist(np.asarray([particle.state[:2] for particle in self.particle_list])))
        ind1, ind2 = np.where(D <= 2 * PARTICLE_RADIUS)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):
            # mass
            m1 = self.particle_list[i1].mass
            m2 = self.particle_list[i2].mass

            # location vector
            r1 = self.particle_list[i1].state[:2]
            r2 = self.particle_list[i2].state[:2]

            # velocity vector
            v1 = self.particle_list[i1].state[2:]
            v2 = self.particle_list[i2].state[2:]

            # relative location & velocity vectors
            r_rel = r1 - r2
            v_rel = v1 - v2

            # momentum vector of the center of mass
            v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)

            # collisions of spheres reflect v_rel over r_rel
            rr_rel = np.dot(r_rel, r_rel)
            vr_rel = np.dot(v_rel, r_rel)
            v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

            # assign new velocities
            self.particle_list[i1].state[2:] = v_cm + v_rel * m2 / (m1 + m2)
            self.particle_list[i2].state[2:] = v_cm - v_rel * m1 / (m1 + m2) 

        for particle in self.particle_list:
            if particle.state[0] < particle.radius + self.bounds[0]:
                particle.state[0] = particle.radius + self.bounds[0]
                particle.state[2] *= -1
            if particle.state[0] > -particle.radius + self.bounds[1]:
                particle.state[0] = -particle.radius + self.bounds[1]
                particle.state[2] *= -1
            if particle.state[1] < particle.radius + self.bounds[2]:
                particle.state[1] = particle.radius + self.bounds[2]
                particle.state[3] *= -1
            if particle.state[1] > -particle.radius + self.bounds[3]:
                particle.state[1] = -particle.radius + self.bounds[3]
                particle.state[3] *= -1

            particle.state[:2] += dt*particle.state[2:]


# Set up figure
fig = plt.figure()
fig.suptitle('Ideal Gas In a Box')
ax_1 = fig.add_axes([0.05,0.2,0.5,0.65], xlim=(X_LOWER, X_UPPER), 
                    ylim=(Y_LOWER, Y_UPPER))

time_display = ax_1.text(0.12, 0.05, '', transform=ax_1.transAxes)

ax_2 = fig.add_axes([0.6, 0.45,0.35,0.2])


offset = 0.06 # an offset to make particles look like they are actually bouncing off walls
# TODO: link marker size to box collisions and ensure that regardless which parameter you change,
#       the wall collision behaviour is unchanged (ie balls still look like they hit walls)
rect = plt.Rectangle((-0.5*BOX_WIDTH-offset, -0.5*BOX_HEIGHT-offset), BOX_WIDTH+2*offset, 
                     BOX_HEIGHT+2*offset, lw=1.5, ec='k', fc='none')

ax_1.add_patch(rect)

np.random.seed(0)
blue_box = ParticleBox(100, 'b', [-0.5*BOX_WIDTH, 0.5*BOX_WIDTH, -0.5*BOX_HEIGHT, 0.5*BOX_HEIGHT])
red_box = ParticleBox(1, 'r', [-0.5*BOX_WIDTH, 0.5*BOX_WIDTH, -0.5*BOX_HEIGHT, 0.5*BOX_HEIGHT])

#TODO: find correct conversion factor for radius to marker size (in points)
marker_size = int(fig.dpi*2*PARTICLE_RADIUS*fig.get_figwidth()
                / np.diff(ax_1.get_xbound())[0])

blue_particle_pos, = ax_1.plot([], [],'bo', markersize=marker_size)
red_particle_pos, = ax_1.plot([], [], 'ro', markersize=marker_size)

dt = 1./30

def init():
    blue_particle_pos.set_data([], [])
    red_particle_pos.set_data([], [])
    time_display.set_text('')
    return blue_particle_pos, red_particle_pos, time_display

def animate(i):
    global blue_box, red_box, dt
    blue_box.step(dt)
    red_box.step(dt)
    
    blue_particle_pos_x = [particle.state[0] for particle in blue_box.particle_list]
    blue_particle_pos_y = [particle.state[1] for particle in blue_box.particle_list]
    
    red_particle_pos_x = [particle.state[0] for particle in red_box.particle_list]
    red_particle_pos_y = [particle.state[1] for particle in red_box.particle_list]

    blue_particle_pos.set_data(blue_particle_pos_x, blue_particle_pos_y)
    red_particle_pos.set_data(red_particle_pos_x, red_particle_pos_y)
    time_display.set_text('time = %.1fs' % blue_box.t)
    return blue_particle_pos, red_particle_pos, time_display

anim = animation.FuncAnimation(fig, animate, frames=30, interval=30, blit=True,
                            init_func=init)

plt.show()


        



