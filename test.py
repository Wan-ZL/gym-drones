import time
import argparse
import numpy as np

# from gym_pybullet_drones.envs.BaseAviary import BaseAviary, DroneModel, Physics
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool


# Note: adjust debug camera target point by edit resetDebugVisualizerCamera in 'BaseAviary.py'
# debug camera keymap, see: https://github.com/robotlearn/pyrobolearn/blob/master/docs/pybullet.md

def stepToTarget(preXYZ: np.array, tarXYZ: np.array, control_freq_hz: int) -> list:
    return_XYZ = (tarXYZ - preXYZ) / control_freq_hz + preXYZ
    return return_XYZ



if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=5,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=False,      type=str2bool,      help='Whether to add obstacles to the environment (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=12,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()


    #### Initialize the simulation #############################
    H = 0.1
    H_STEP = .05
    R = .3

    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(ARGS.num_drones)])

    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/ARGS.num_drones] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    env = CtrlAviary(drone_model=ARGS.drone,
                     num_drones=ARGS.num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=ARGS.physics,
                     neighbourhood_radius=10,
                     freq=ARGS.simulation_freq_hz,
                     aggregate_phy_steps=AGGR_PHY_STEPS,
                     gui=ARGS.gui,
                     record=ARGS.record_video,
                     obstacles=ARGS.obstacles,
                     user_debug_gui=ARGS.user_debug_gui
                     )

    PYB_CLIENT = env.getPyBulletClient()
    PERIOD = 10
    NUM_WP = ARGS.control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(ARGS.num_drones)])

    #### Initialize the controllers ############################
    ctrl = [DSLPIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]
    # ctrl = [SimplePIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]


    #### Run the simulation ####################################
    #### (0,0) is base station ####
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    print("CTRL_EVERY_N_STEPS", CTRL_EVERY_N_STEPS)
    action = {str(i): np.array([12713,12713,12713,12713]) for i in range(ARGS.num_drones)}
    START = time.time()
    frameN = 1

    # initial target scan area/map
    map_x = 1   # original point of target area
    map_y = 1
    map_size = 5
    map_border = 1  # create a boarder to avoid 'index error'
    map_x_index_const = map_x - map_border
    map_y_index_const = map_y - map_border
    map_size_with_border = map_size + (2 * map_border)
    scan_map = np.zeros((map_size_with_border,map_size_with_border))
    # index_const = int(map_size_with_border/2)

    # initial position for drones
    Desti_XYZ = INIT_XYZS
    for i in range(ARGS.num_drones):
        Desti_XYZ[i] = INIT_XYZS[i]
    TARG_XYZS = Desti_XYZ
    PRE_XYZ = Desti_XYZ

    # path planning
    drones_path = {}
    for id in range(ARGS.num_drones):
        x_temp_set = np.zeros(map_size).reshape(map_size,1) + map_x + id
        y_temp_set = np.arange(map_size).reshape(map_size,1) + map_y
        z_temp_set = np.random.uniform(2, 2, map_size).reshape(map_size,1)
        drones_path[id] = np.concatenate((x_temp_set, y_temp_set), axis = 1)
        drones_path[id] = np.concatenate((drones_path[id], z_temp_set), axis=1)

    print("drones_path", drones_path)

    while True:
        # update destinatin for drones
        if frameN % 500 == 0:
            for i in range(ARGS.num_drones):
                # print(drones_path[i])
                # Desti_XYZ[i] = np.array([np.random.uniform(map_x, map_x + map_size), np.random.uniform(map_y, map_y + map_size), np.random.uniform(0, 2)])
                Desti_XYZ[i] = drones_path[i][0]
                if drones_path[i].shape[0] > 1:
                    drones_path[i] = drones_path[i][1:,:]
                # print("Desti_XYZ: ", i, Desti_XYZ[i])
            print("scan_map \n", scan_map)
        TARG_XYZS = stepToTarget(PRE_XYZ, Desti_XYZ, ARGS.control_freq_hz)

        PRE_XYZ = TARG_XYZS




        H_STEP = .05
        R = .3
        Tar_H = np.random.uniform(0,1,1)
        # TARG_XYZS = np.array([[R * np.cos((frameN / 6) * 2 * np.pi + np.pi / 2), R * np.sin((frameN / 6) * 2 * np.pi + np.pi / 2) - R,
        #                        Tar_H + frameN * H_STEP] for frameN in range(ARGS.num_drones)])

        # action = {str(frameN): np.array([1, 0, 0, 0]) for frameN in range(ARGS.num_drones)}
        # obs, reward, done, info = env.step(env.action_space.sample())
        obs, reward, done, info = env.step(action)

        # count cell scan by frame
        for i in range(ARGS.num_drones):
            cell_x, cell_y = obs[str(i)]["state"][0:2].astype(int)

            map_x_index = cell_x - map_x_index_const
            map_y_index = cell_y - map_y_index_const

            if 0 <= map_x_index and map_x_index <= map_size_with_border and 0 <= map_y_index and map_y_index <= map_size_with_border:
                scan_map[map_x_index, map_y_index] += 0.01
            else:
                print("cell_x, cell_y", i, cell_x, cell_y)
                print("map_x_index_const, map_y_index_const", i, map_x_index, map_y_index)


        for i in range(ARGS.num_drones):
            # print(i)
            # print("TARG_XYZS", TARG_XYZS[i])
            # print("drone pos:", obs[str(i)]["state"][0:3])
            # action[str(i)], _, _ = ctrl[i].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
            #                                                        state=obs[str(i)]["state"],
            #                                                        target_pos=TARG_XYZS,
            #                                                        # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
            #                                                        target_rpy=INIT_RPYS[i, :]
            #                                                        )
            action[str(i)], _, _ = ctrl[i].computeControlFromState(control_timestep = CTRL_EVERY_N_STEPS * env.TIMESTEP,
                                                                   state = obs[str(i)]["state"],
                                                                   target_pos = TARG_XYZS[i])
            # action[str(i)], _, _ = ctrl[i].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
            #                                                        state=obs[str(i)]["state"],
            #                                                        target_pos=np.hstack([TARGET_POS[wp_counters[i], 0:2], INIT_XYZS[i, 2]]),
            #                                                        # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
            #                                                        target_rpy=INIT_RPYS[i, :]
            #                                                        )

        # env.render()
        # if ARGS.gui:
        #     sync(frameN, START, env.TIMESTEP)
        frameN += 1



