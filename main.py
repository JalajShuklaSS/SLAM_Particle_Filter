# Pratik Chaudhari (pratikac@seas.upenn.edu)

import click, tqdm, random

from slam import *

def run_dynamics_step(src_dir, log_dir, idx, split, t0=0, draw_fig=False):
    """
    This function is for testing the dynamics update step. It creates two figures: 
    one showing the robot location trajectory using odometry information obtained from the lidar,
    and the other showing the trajectory using the PF with a very small dynamics noise.
    """
    slam = slam_t(Q=1e-5 * np.eye(3))
    slam.read_data(src_dir, idx, split)

    # Trajectory using odometry
    d = slam.lidar
    xyth = []
    for p in d:
        xyth.append([p['xyth'][0], p['xyth'][1], p['xyth'][2]])
    xyth = np.array(xyth)

    plt.figure(1)
    plt.clf()
    plt.title('Trajectory using onboard odometry')
    plt.plot(xyth[:, 0], xyth[:, 1], color='blue')  # Light blue color
    logging.info('> Saving odometry plot in ' + os.path.join(log_dir, 'odometry_%s_%02d.jpg' % (split, idx)))
    plt.savefig(os.path.join(log_dir, 'odometry_%s_%02d.jpg' % (split, idx)))

    # Dynamics propagation using particle filter
    n = 3
    w = np.ones(n) / float(n)
    p = np.zeros((3, n), dtype=np.float64)
    slam.init_particles(n, p, w)
    slam.p[:, 0] = deepcopy(slam.lidar[0]['xyth'])

    print('> Running prediction')
    t0 = 0
    T = len(d)
    ps = deepcopy(slam.p)  # maintains all particles across all time steps
    plt.figure(2, figsize=(12, 12))  # Increase size of the plot
    plt.clf()
    ax = plt.subplot(111)
    for t in tqdm.tqdm(range(t0 + 1, T)):
        slam.dynamics_step(t)
        ps = np.hstack((ps, slam.p))

        if draw_fig:
            ax.clear()
            ax.plot(slam.p[0], slam.p[0], '*r')
            plt.title('Particles %03d' % t)
            plt.draw()
            plt.pause(0.01)

    plt.plot(ps[0], ps[1], '*c')
    plt.title('Trajectory using PF')
    logging.info('> Saving plot in ' + os.path.join(log_dir, 'dynamics_only_%s_%02d.jpg' % (split, idx)))
    plt.savefig(os.path.join(src_dir, log_dir, 'dynamics_only_%s_%02d.jpg' % (split, idx)))

def run_observation_step(src_dir, log_dir, idx, split, is_online=False):
    """
    This function is for debugging the observation update step.
    It initializes three particles and updates them for one time step.
    """
    slam = slam_t(resolution=0.05)
    slam.read_data(src_dir, idx, split)

    # Initialize particles
    slam.init_particles(n=3, p=np.array([[0.2, 2, 3], [0.4, 2, 5], [0.1, 2.7, 4]]).T, w=np.ones(3))

    slam.observation_step(t=0)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

    # Reinitialize particles
    n = 3
    w = np.ones(n) / float(n)
    p = np.array([[2, 0.2, 3], [2, 0.4, 5], [2.7, 0.1, 4]])
    slam.init_particles(n, p.T, w)

    slam.observation_step(t=1)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))


def plotting(slam, particles, estimate):
    gmap = np.array(slam.map.cells).astype(float)
    gmap[slam.map.num_obs_per_cell != 0] += 0.2
    gmap = np.dstack((gmap, gmap))
    gmap = np.dstack((gmap, np.array(slam.map.cells))) * 200
    gmap[slam.map.cells == 1] = [255, 255, 255]
    gmap[estimate[0, :], estimate[1, :]] = [173, 216, 230]  # Light blue color for estimate

    # Create the 'logs' directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    plt.imshow(gmap.astype(int), origin="lower")
    plt.plot(particles[:, 0], particles[:, 1], '.r', markersize=5)  # Plot red particles
    plt.xlim([0, slam.map.szx])
    plt.ylim([0, slam.map.szy])
    plt.title('Map')
    plt.savefig(os.path.join('logs', f'{estimate.shape[1]}.png'))
    plt.close()
    
def Odometry_data(slam, t, od=0):
    if od:
        xyth = slam.lidar[t]['xyth']
        return slam.map.grid_cell_from_xy(xyth[0], xyth[1]).reshape(-1, 1)
    else:
        return slam.map.grid_cell_from_xy(slam.instantaneous[0], slam.instantaneous[1]).reshape(-1, 1)


def run_slam(src_dir, log_dir, idx, split):
    """
    This function runs slam.
    """
    slam = slam_t(resolution=0.05, Q=np.diag([2e-4, 2e-4, 1e-4]))
    slam.read_data(src_dir, idx, split)
    t0 = 0
    t0 = slam.find_joint_t_idx_from_lidar(t0)

    xyth = slam.lidar[t0]['xyth']
    xyth[2] = slam.lidar[t0]['rpy'][2]
    slam.init_particles(n=1, p=xyth.reshape((3, 1)), w=np.array([1]))
    slam.observation_step(t=t0)

    sync = slam.lidar[t0]['t']
    iteration = 0
    slam.init_particles(n=100, t0=t0)
    od = slam.map.grid_cell_from_xy(xyth[0], xyth[1]).reshape(-1, 1)
    for t in tqdm.tqdm(range(t0, len(slam.lidar))):
        try:
            nod = Odometry_data(slam, slam.find_joint_t_idx_from_lidar(t + sync), od=0)
            od = np.column_stack((od, nod))
        except:
            pass
    estimate = slam.map.grid_cell_from_xy(slam.instantaneous[0], slam.instantaneous[1]).reshape(-1, 1)
    particles = []  # Store particles for plotting red line
    for t in tqdm.tqdm(range(t0, len(slam.lidar))):
        slam.dynamics_step(t)
        slam.observation_step(t)
        final = Odometry_data(slam, t, od=0)
        estimate = np.column_stack((estimate, final))
        particles.append(slam.instantaneous[:2])  # Store current particle for plotting red line

        if iteration % 100 == 0 and iteration > 2000:
            plotting(slam, np.array(particles), estimate)
            plt.pause(1e-10)
        iteration += 1
    plt.savefig(os.path.join(src_dir, log_dir, 'map_%s_%02d.jpg' % (split, idx)))


@click.command()
@click.option('--src_dir', default='C:/D/2024/Spring Sem courses/ESE 650 Learning in robo/hw3/hw_export/hw3/p2/p2/p2',
              help='data directory', type=str)
@click.option('--log_dir', default='logs', help='directory to save logs', type=str)
@click.option('--idx', default='1', help='dataset number', type=int)
@click.option('--split', default='train', help='train/test split', type=str)
@click.option('--mode', default='slam',
              help='choices: dynamics OR observation OR slam', type=str)
def main(src_dir, log_dir, idx, split, mode):
    # Run python main.py --help to see how to provide command line arguments

    if not mode in ['slam', 'dynamics', 'observation']:
        raise ValueError('Unknown argument --mode %s' % mode)
        sys.exit(1)

    np.random.seed(42)
    random.seed(42)

    if mode == 'dynamics':
        run_dynamics_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    elif mode == 'observation':
        run_observation_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    else:
        p = run_slam(src_dir, log_dir, idx, split)
        return p


if __name__ == '__main__':
    main()