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
SCALE_FACTOR = 0.5
BOX_WIDTH = SCALE_FACTOR*(X_UPPER-X_LOWER)
BOX_HEIGHT = SCALE_FACTOR*(Y_UPPER-Y_LOWER)

class GasParticle:
    """docstring for GasParticle."""

    """To understand how we are going to initialize our ball object, 
    we need to understand how to plot artists in matplotlib"""
    def __init__(self, mass = PARTICLE_MASS, radius = PARTICLE_RADIUS,
                 init_state = [0.0, 0.0, 0.0, 0.0]):
        self.mass = mass
        self.radius = radius
        self.init_state = np.asarray(init_state, dtype=float)
        self.state = self.init_state.copy()

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
    def __init__(self, N=25, bounds=[-1.0, 1.0, -1.0, 1.0]):
        self.t = 0.0
        self.N = N
        self.bounds = bounds
        self.particle_list = np.array([GasParticle(PARTICLE_MASS, PARTICLE_RADIUS,
                                        [-0.5*BOX_WIDTH + BOX_WIDTH*np.random.random(), 
                                        -0.5*BOX_HEIGHT + BOX_HEIGHT*np.random.random(),
                                        -1.0 + np.random.random(), -1.0 + np.random.random()]) 
                                        for i in np.arange(N)])

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
my_box = ParticleBox(2, [-0.5*BOX_WIDTH, 0.5*BOX_WIDTH, -0.5*BOX_HEIGHT, 0.5*BOX_HEIGHT])

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
    
    particle_pos_x = [particle.state[0] for particle in my_box.particle_list]
    particle_pos_y = [particle.state[1] for particle in my_box.particle_list]
    
    particle_pos.set_data(particle_pos_x, particle_pos_y)
    time_display.set_text('time = %.1fs' % my_box.t)
    return particle_pos, time_display

anim = animation.FuncAnimation(fig, animate, frames=30, interval=10, blit=True,
                               init_func=init)

plt.show()        



