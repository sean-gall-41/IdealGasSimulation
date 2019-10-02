import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation

# GLOBAL VARIABLES

X_LOWER, X_UPPER, Y_LOWER, Y_UPPER = -6.0, 6.0, -6.0, 6.0
PARTICLE_MASS, PARTICLE_RADIUS = 1.2E-20, 0.12
SCALE_FACTOR = 0.75
BOX_WIDTH = SCALE_FACTOR * (X_UPPER - X_LOWER)
BOX_HEIGHT = SCALE_FACTOR * (Y_UPPER - Y_LOWER)
# NOTE: if we are going to use SI units, have to be consistent throughout,
# so if I give Boltzmann's constant as 1.38064852E-23 m^2kgs^-2K^- then
# everything else must match.
KB = 1.38064852E-23


def randomReal(a, b):
    """returns a randomly distributed real number in the
       half open interval [a, b)"""
    return (b - a) * np.random.random() + a


def lengths(coord):
    """Given a numpy array of coordinates (in our case with
       shape (N, 2) return an array of the magnitude of the
       vectors the coordinates represent, in the usual Euclidean
       metric"""
    return np.sqrt(coord[:, 0]**2 + coord[:, 1]**2)


def MaxWell2D(data, a, b):
    return a * data * np.exp(b * data**2)


class Error(Exception):
    """Base class for other custom-defined exceptions (if the need arises)"""
    pass


class OutOfBoundsError(Error):
    """Raised when something is out of bounds (like a particle)"""
    pass


class GasParticle:
    """docstring for GasParticle."""

    def __init__(self, mass=PARTICLE_MASS, radius=PARTICLE_RADIUS,
                 color='b', init_state=[0.0, 0.0, 0.0, 0.0]):
        self.mass = mass
        self.radius = radius
        self.color = color
        self.init_state = np.asarray(init_state, dtype=float)
        self.state = self.init_state.copy()

    # debating whether to define equality in terms of intrinsic properties
    # like mass, radius, and color, rather than state, or include state.
    def __eq__(self, other):
        return True if self.mass == other.mass and \
            self.radius == other.radius and \
            self.color == other.color and \
            np.array_equal(self.state, other.state) else False

    # predicate function to determine whether the caller collides with
    # the parameter particle
    def collides_with(self, other):
        D = np.sqrt((self.state[0] - other.state[0])**2 +
                    (self.state[1] - other.state[1])**2)

        return True if D <= self.radius + other.radius else False


# A particle box represents a group of N gas particles confined
# within a box of dimensions bounds
class ParticleBox():
    def __init__(self, N=25, color='b', bounds=[-1.0, 1.0, -1.0, 1.0]):
        self.t = 0.0
        self.N = N
        self.bounds = bounds
        self.particle_list = np.array(
            [GasParticle(PARTICLE_MASS, PARTICLE_RADIUS, color, [
                randomReal(-0.5 * BOX_WIDTH, 0.5 * BOX_WIDTH),
                randomReal(-0.5 * BOX_HEIGHT, 0.5 * BOX_HEIGHT),
                randomReal(-5, 5), randomReal(-5, 5)]) for i in np.arange(N)])

    def add_particle(self, particle):
        if particle.state[0] - particle.radius < self.bounds[0] or \
           particle.state[0] + particle.radius > self.bounds[1] or \
           particle.state[1] - particle.radius < self.bounds[2] or \
           particle.state[1] + particle.radius > self.bounds[3]:
            raise OutOfBoundsError
        self.N += 1
        self.particle_list = np.append(self.particle_list, [particle])

    # Steps the state of the particle box forward by time dt
    # => all particle positions and velocities are updated.
    def step(self, dt):
        # Select only those particles undergoing collisions
        D = squareform(pdist(np.asarray([particle.state[:2] for particle in
                       self.particle_list])))
        ind1, ind2 = np.where(D <= 2 * PARTICLE_RADIUS)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):
            # mass
            m1 = self.particle_list[i1].mass
            m2 = self.particle_list[i2].mass

            # location vector (time delta_t after contact)
            r1 = self.particle_list[i1].state[:2]
            r2 = self.particle_list[i2].state[:2]

            # velocity vector (time delta_t after contact)
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
        # TODO: re-write bounds checks to fit above loop variables so
        # as to fit everything within one loop (though it is no less efficient
        # do this for readability

        # bounds check to ensure particles bounce off walls
        for particle in self.particle_list:
            if particle.state[0] <= particle.radius + self.bounds[0]:
                particle.state[0] = particle.radius + self.bounds[0]
                particle.state[2] *= -1
            if particle.state[0] >= -particle.radius + self.bounds[1]:
                particle.state[0] = -particle.radius + self.bounds[1]
                particle.state[2] *= -1
            if particle.state[1] <= particle.radius + self.bounds[2]:
                particle.state[1] = particle.radius + self.bounds[2]
                particle.state[3] *= -1
            if particle.state[1] >= -particle.radius + self.bounds[3]:
                particle.state[1] = -particle.radius + self.bounds[3]
                particle.state[3] *= -1
            # update particle positions after collisions with walls
            particle.state[:2] += dt * particle.state[2:]
        # Update internal clock
        self.t += dt


# Set up figure
fig = plt.figure()
fig.suptitle('Ideal Gas In a Box')
ax_1 = fig.add_axes([0.05, 0.2, 0.5, 0.65], xlim=(X_LOWER, X_UPPER),
                    ylim=(Y_LOWER, Y_UPPER))
ax_1.set_xlabel('$x$')
ax_1.set_ylabel('$y$')
# set up simulation time dispaly
time_display = ax_1.text(0.12, 0.05, '', transform=ax_1.transAxes)

# an offset to make particles look like they are actually bouncing off walls
offset = 0.1
rect = plt.Rectangle((-0.5 * BOX_WIDTH - offset, -0.5 * BOX_HEIGHT - offset),
                     BOX_WIDTH + 2 * offset, BOX_HEIGHT + 2 * offset, lw=1.5,
                     ec='k', fc='none')

ax_1.add_patch(rect)

# initialize random seed for consistent results for debug
np.random.seed(0)
N = 99
particle_box = ParticleBox(N, 'b', [-0.5 * BOX_WIDTH, 0.5 * BOX_WIDTH,
                                    -0.5 * BOX_HEIGHT, 0.5 * BOX_HEIGHT])

try:
    # add special particle to our box for debugging as well as future
    # studies of Brownian drift perhaps
    special_particle_radius = PARTICLE_RADIUS
    special_particle = GasParticle(PARTICLE_MASS,
                                   special_particle_radius,
                                   'r',
                                   [randomReal(-0.5 * BOX_WIDTH,
                                               0.5 * BOX_WIDTH),
                                    randomReal(-0.5 * BOX_HEIGHT,
                                               0.5 * BOX_HEIGHT),
                                    randomReal(-5, 5),
                                    randomReal(-5, 5)])
    particle_box.add_particle(special_particle)

    # add an axis to the figure for the velocity distribution plots
    ax_2 = fig.add_axes([0.6, 0.45, 0.35, 0.2])
    ax_2.set_xlabel('$v$')
    ax_2.set_ylabel('$dN$')
    # obtain velocity components for all N particles
    velocity_components = np.array(
        [particle.state[2:] for particle in particle_box.particle_list])
    # Create array of speeds of particles
    velocities = lengths(velocity_components)
    # Assume total energy of the system is conserved, calculate based
    # off initial velocities
    total_energy = 0.5 * PARTICLE_MASS * sum(velocities**2)
    # NOTE: total_energy = 85.98714633280777 with seed 0
    beta = N / total_energy  # from ideal gas energy temperature relation

    # velocities range to plot theoretical maxwell dist against
    veloc_range = np.linspace(0.0, max(velocities), 500)

    # TODO: fix theoretic calculation of maxwell dist for 2D
    # NOTE: number of particles per bin depends on the number of bins whereas
    # the theoretical distribution has no dependence on number of bins...
    boltzmann_dist = N * beta * PARTICLE_MASS * veloc_range * \
        np.exp(-0.5 * PARTICLE_MASS * beta * veloc_range**2) * \
        (max(veloc_range)-min(veloc_range)) / len(veloc_range)

    # deltaV = (max(veloc_range)-min(veloc_range)) / len(veloc_range)
    # a = beta * PARTICLE_MASS * deltaV
    # b = -0.5 * PARTICLE_MASS * beta

    # Plot the theoretical distribution
    ax_2.plot(veloc_range, boltzmann_dist, 'b-', label='theoretical dist')

    # create the histogram values and bins for experimental velocity dist
    n, bins = np.histogram(velocities, N)
    # obtain list of bin centers in order to fit the above rayleigh dist
    # for x and y of the same size
    bin_centers = np.array([
        0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    params, params_covariance = optimize.curve_fit(MaxWell2D, bin_centers, n,
                                                   p0=[0.05 * 1.E-1, -5.E-2])
    # plot the resultant fit TODO: fix the model and maybe initial guess,
    # obtaining poor fit
    active_curve_fit, = ax_2.plot([], [], 'r-', label='experimental fit')
    # Going to histogram probability to be in a velocity range, rather than
    # histogram number of particles in a velocity range

    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n
    nrects = len(left)

    # Now indicate where the bounds of the rectangles are
    nverts = nrects * (1 + 3 + 1)
    verts = np.zeros((nverts, 2))
    codes = np.ones(nverts, int) * path.Path.LINETO
    codes[0::5] = path.Path.MOVETO
    codes[4::5] = path.Path.CLOSEPOLY
    verts[0::5, 0] = left
    verts[0::5, 1] = bottom
    verts[1::5, 0] = left
    verts[1::5, 1] = top
    verts[2::5, 0] = right
    verts[2::5, 1] = top
    verts[3::5, 0] = right
    verts[3::5, 1] = bottom

    # TODO: find correct conversion factor for radius to marker size (in
    # points)
    marker_size = int(fig.dpi * 2 * PARTICLE_RADIUS * fig.get_figwidth() /
                      np.diff(ax_1.get_xbound())[0])-8
    blue_particle_pos, = ax_1.plot([], [], 'bo', markersize=marker_size)
    red_particle_pos, = ax_1.plot([], [], 'ro', markersize=marker_size)

    # Time interval of one second
    dt = 1. / 30

    # def init():
    #     blue_particle_pos.set_data([], [])
    #     red_particle_pos.set_data([], [])
    #     time_display.set_text('')
    #     return blue_particle_pos, red_particle_pos, time_display

    # Patch will later be a path patch
    patch = None

    # the function called for FuncAnimate. In charge of stepping simulation
    # forward one time-step, which involves calculating new positions and
    # velocities, plotting positions, calculating speeds, then histogramming
    # them
    def animate(i):
        global particle_box, dt, velocity_components, velocities, veloc_range
        particle_box.step(dt)
        # red_box.step(dt)

        blue_particle_pos_x = [particle_box.particle_list[i].state[0] for i in
                               np.arange(particle_box.particle_list.size - 1)]
        blue_particle_pos_y = [particle_box.particle_list[i].state[1] for i in
                               np.arange(particle_box.particle_list.size - 1)]

        red_particle_pos_x = [
            particle_box.particle_list[
                particle_box.particle_list.size - 1].state[0]]
        red_particle_pos_y = [
            particle_box.particle_list[
                particle_box.particle_list.size - 1].state[1]]

        blue_particle_pos.set_data(blue_particle_pos_x, blue_particle_pos_y)
        red_particle_pos.set_data(red_particle_pos_x, red_particle_pos_y)
        time_display.set_text('time = %.1fs' % particle_box.t)

        velocity_components = np.array(
            [particle.state[2:] for particle in particle_box.particle_list])
        velocities = lengths(velocity_components)

        n, bins = np.histogram(velocities, N)
        top = bottom + n
        verts[1::5, 1] = top
        verts[2::5, 1] = top

        if i % 10 == 0:
            params, params_covariance = optimize.curve_fit(
                MaxWell2D, bin_centers, n, p0=[0.05 * 1.E-1, -5.E-2])
            active_curve_fit.set_data(
                veloc_range, MaxWell2D(veloc_range, params[0], params[1]))

        return blue_particle_pos, red_particle_pos, time_display,\
            active_curve_fit, patch

    # set up histogram bars, limits on histogram plot
    barpath = path.Path(verts, codes)
    patch = patches.PathPatch(barpath, facecolor='green', edgecolor='yellow',
                              alpha=0.5, label='experimental dist')
    ax_2.add_patch(patch)
    ax_2.set_xlim(left[0], right[-1])
    ax_2.set_ylim(bottom.min(), top.max())

    ax_2.legend()

    anim = animation.FuncAnimation(
        fig, animate, frames=30, interval=30, blit=True)
    """,init_func=init)"""

    plt.show()

except OutOfBoundsError:
    print('Special particle is out of bounds!')
