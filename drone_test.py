import random
import time
import argparse
import numpy as np

from model_System import system_model
from model_Defender import defender_model
from model_Attacker import attacker_model


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

# def stepToTarget(preXYZ: np.array, Desti_XYZ: np.array, control_freq_hz: int) -> list:
#     return_XYZ = (Desti_XYZ - preXYZ) / control_freq_hz + preXYZ
#     return return_XYZ
def updateTempLoca(drone_set, control_freq_hz: int):
    for drone in drone_set:
        drone.xyz_temp = (drone.xyz - drone.xyz_temp) / control_freq_hz + drone.xyz_temp
    # return_XYZ = (Desti_XYZ - preXYZ) / control_freq_hz + preXYZ
    # return return_XYZ

# distance between two drones. (Eq. \eqref(Eq: distance) in paper)
def drones_distance(drone_x_location, drone_y_location):
    distance_squre = np.square(drone_x_location - drone_y_location)
    return np.sqrt(np.sum(distance_squre))



if __name__ == "__main__":
    # create model class
    system = system_model()
    defender = defender_model(system)
    attacker = attacker_model(system)
    MD_dict = system.MD_dict
    HD_dict = system.HD_dict
    MD_set = MD_dict.values()
    HD_set = HD_dict.values()



    # sample for obtain general parameter of MD and HD
    HD_sample = system.sample_HD()
    MD_sample = system.sample_MD()



    #### default parameters:
    num_MD = system.num_MD # number of MD (in index, MD first then HD)
    num_HD = system.num_HD  # number of HD
    maximum_signal_radius_HD = HD_sample.maximum_signal_radius    # signal radius of HD
    sg_HD = defender.sg_HD # defense strategy, range [0, 10]
    min_sg_HD = defender.min_sg_HD   # minimum signal level defender can choose
    max_sg_HD = defender.max_sg_HD  # maximum signal level defender can choose
    tao_lower = defender.tao_lower   # The lower bounds of the number of MDs that HDs can protect simultaneously
    tao_upper = defender.tao_upper   # The upper bounds of the number of MDs that HDs can protect simultaneously
    target_map_size = system.target_map_size # size of surveillance area (map size)


    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2p",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=num_HD + num_MD, type=int, help='Number of drones', metavar='')
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
    H = 1
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
    map_x = system.map_ori_x   # original point of target area
    map_y = system.map_ori_y

    map_border = 0  # create a boarder to avoid 'index error'
    map_x_index_const = map_x - map_border
    map_y_index_const = map_y - map_border
    map_size_with_border = target_map_size+1 + (2 * map_border)
    print("map_size_with_border", map_size_with_border)
    scan_map = np.zeros((map_size_with_border,map_size_with_border))
    # index_const = int(map_size_with_border/2)

    # initial position for drones (MD+HD)
    for MD in MD_set:
        MD.assign_destination(INIT_XYZS[MD.ID])
        # MD.xyz = INIT_XYZS[MD.ID]
        MD.xyz_temp = MD.xyz
    for HD in HD_set:
        HD.assign_destination(INIT_XYZS[num_MD+HD.ID])
        # HD.xyz = INIT_XYZS[num_MD+HD.ID]
        HD.xyz_temp = HD.xyz

    # Desti_XYZ = INIT_XYZS
    # for i in range(ARGS.num_drones):
    #     Desti_XYZ[i] = INIT_XYZS[i]
    # TARG_XYZS = Desti_XYZ
    # PRE_XYZ = Desti_XYZ

    # path planning (test)
    # MD_trajectory = {}
    # for id in range(ARGS.num_drones):
    #     x_temp_set = np.zeros(target_map_size).reshape(target_map_size,1) + map_x + id
    #     y_temp_set = np.arange(target_map_size).reshape(target_map_size,1) + map_y
    #     z_temp_set = np.random.uniform(2, 2, target_map_size).reshape(target_map_size,1)
    #     MD_trajectory[id] = np.concatenate((x_temp_set, y_temp_set), axis = 1)
    #     MD_trajectory[id] = np.concatenate((MD_trajectory[id], z_temp_set), axis=1)

    # path planning for MD
    MD_trajectory = defender.MD_trajectory
    # MD_trajectory = defender.MD_trajectory(num_MD, target_map_size)
    # MD_trajectory = defender.create_trajectory()

    # Z height for MD
    # z_range_start_MD = 1
    # z_range_end_MD = 2
    # if num_MD == 1:
    #     z_mutiplier = z_range_end_MD
    # else:
    #     z_mutiplier = (z_range_end_MD - z_range_start_MD) / (num_MD - 1)
    # for id in range(num_MD):
    #     MD_trajectory[id] = np.insert(MD_trajectory[id], 2, z_mutiplier * id + z_range_start_MD, axis=1)

    # path planning for HD
    drones_deploy_HD = {}
    for HD in HD_set:
        drones_deploy_HD[HD.ID] = np.zeros(3) # this deployment only depends on MD's latest location

    # Z height for HD
    z_range_start_HD = 2.2
    z_range_end_HD = 3
    if num_HD == 1:
        z_mutiplier = z_range_end_HD
    else:
        z_mutiplier = (z_range_end_HD - z_range_start_HD) / (num_HD - 1)
    for HD in HD_set:
        drones_deploy_HD[HD.ID][2] = z_mutiplier * HD.ID + z_range_start_HD



    while system.is_mission_Not_complete():
        # update destination for drones (every 500 frames)
        update_freq = 500
        if frameN % update_freq == 0:

            # check if mission complete
            system.check_mission_complete()

            # path planning for MD
            if defender.is_calc_trajectory():
                MD_trajectory = defender.MD_trajectory

            # check real time scan
            print("map \n", scan_map.round(1))

            # for MD update next destination
            for MD in MD_set:
                MD.assign_destination(MD_trajectory[MD.ID][0])
                # MD.xyz = MD_trajectory[MD.ID][0]
                if MD_trajectory[MD.ID].shape[0] > 1:
                    MD_trajectory[MD.ID] = MD_trajectory[MD.ID][1:, :]
                else:
                    print("MD arrived:", MD.ID)
                    MD.no_mission()

            # for i in range(num_MD):
            #     # print(MD_trajectory[i])
            #     Desti_XYZ[i] = MD_trajectory[i][0]
            #     if MD_trajectory[i].shape[0] > 1:
            #         MD_trajectory[i] = MD_trajectory[i][1:, :]
                # print("Desti_XYZ: ", i, Desti_XYZ[i])


            # avoid HD crash when creating
            if frameN <= update_freq:
                for HD in HD_set:
                    HD.assign_destination(drones_deploy_HD[HD.ID])
                    # HD.xyz = drones_deploy_HD[HD.ID]
                # for i in range(num_MD, num_MD + num_HD):
                #     Desti_XYZ[i] = drones_deploy_HD[i - num_MD]

            # for HD update next destination
            # Algorithm 1 in paper
            L_MD_set = np.arange(num_MD)
            # L_HD_set = np.arange(num_MD, num_MD+num_HD)
            L_HD_set = np.arange(num_HD)
            max_radius = HD_sample.maximum_signal_radius
            p_H_r = (sg_HD * max_radius) / max_sg_HD # actual signal radius under given defense strategy sg_HD
            S_set_HD = {} # A set of HDs with assigned MDs
            for HD_id in L_HD_set:
                if L_MD_set.size == 0:
                    S_set_HD[HD_id] = np.empty(0, dtype=int)
                    continue

                N_l_H_set = np.empty(0) # A set of MDs detected/protected by HD
                for MD_id in L_MD_set:
                    # if drones_distance(Desti_XYZ[MD_id], Desti_XYZ[HD_id]) < p_H_r:
                    if drones_distance(MD_dict[MD_id].xyz, HD_dict[HD_id].xyz) < p_H_r:
                        N_l_H_set = np.append(N_l_H_set, MD_id)

                if N_l_H_set.size < tao_lower:
                    HD_pos_candidate = np.zeros(3)
                    N_l_H_new_set = np.empty(0)
                    for MD_id_candi in L_MD_set:      # search MD position that HD can move to so more MD can be protected
                        temp_set = np.empty(0)
                        for MD_id in L_MD_set:
                            temp_position = MD_dict[MD_id].xyz
                            temp_position[:2] = MD_dict[MD_id].xyz[:2]
                            # if drones_distance(Desti_XYZ[MD_id_candi], Desti_XYZ[MD_id]) < p_H_r:
                            if drones_distance(MD_dict[MD_id_candi].xyz, MD_dict[MD_id].xyz) < p_H_r:
                                temp_set = np.append(temp_set, MD_id)
                        if temp_set.size > N_l_H_new_set.size:  # set new position as candidate
                            N_l_H_new_set = temp_set
                            HD_pos_candidate = MD_dict[MD_id_candi].xyz
                    HD_dict[HD_id].assign_destination_xy(HD_pos_candidate[:2])
                    # HD_dict[HD_id].xyz[:2] = HD_pos_candidate[:2]
                    N_l_H_new_subset = N_l_H_new_set[:tao_upper]
                    L_MD_set = np.delete(L_MD_set, np.searchsorted(L_MD_set, N_l_H_new_subset))    # Remove protected MDs from set L_MD_set
                    S_set_HD[HD_id] = N_l_H_new_subset  # Add deployed HD to set S_set_HD
                elif tao_lower <= N_l_H_set.size and N_l_H_set.size <= tao_upper:
                    L_MD_set = np.delete(L_MD_set, np.searchsorted(L_MD_set,N_l_H_set))     # Remove protected MDs from set L_MD_set
                    S_set_HD[HD_id] = N_l_H_set     # Add deployed HD to set S_set_HD
                else:
                    N_l_H_subset = N_l_H_set[:tao_upper]
                    L_MD_set = np.delete(L_MD_set, np.searchsorted(L_MD_set,N_l_H_subset))     # Remove protected MDs from set L_MD_set
                    S_set_HD[HD_id] = N_l_H_subset  # Add deployed HD to set S_set_HD
                # print("L_MD_set", L_MD_set)
            print("S_set_HD", S_set_HD)

            # print("Desti_XYZ", Desti_XYZ)
            # for MD in MD_set:
            #     Desti_XYZ[MD.ID] = MD.xyz
            # for HD in HD_set:
            #     Desti_XYZ[HD.ID+num_MD] = HD.xyz


        # TARG_XYZS = stepToTarget(TARG_XYZS, Desti_XYZ, ARGS.control_freq_hz)
        updateTempLoca(MD_set, ARGS.control_freq_hz)
        updateTempLoca(HD_set, ARGS.control_freq_hz)


        H_STEP = .05
        R = .3
        Tar_H = np.random.uniform(0,1,1)
        # TARG_XYZS = np.array([[R * np.cos((frameN / 6) * 2 * np.pi + np.pi / 2), R * np.sin((frameN / 6) * 2 * np.pi + np.pi / 2) - R,
        #                        Tar_H + frameN * H_STEP] for frameN in range(ARGS.num_drones)])

        # action = {str(frameN): np.array([1, 0, 0, 0]) for frameN in range(ARGS.num_drones)}
        # obs, reward, done, info = env.step(env.action_space.sample())
        obs, reward, done, info = env.step(action)

        # count cell scan by frame
        for i in range(num_MD):
            # print("obs[str(i)][state]", obs[str(i)]["state"][0:3].round(1))
            cell_x, cell_y, height_z = obs[str(i)]["state"][0:3]

            if height_z < 0.1:
                print("Drone crashed, ID:", i)
                print("cell_x, cell_y, height_z", cell_x, cell_y, height_z)

            map_x_index = int(cell_x + 0.5 + map_border)
            map_y_index = int(cell_y + 0.5 + map_border)

            if 0 <= map_x_index and map_x_index < map_size_with_border and 0 <= map_y_index and map_y_index < map_size_with_border:
                scan_map[map_x_index, map_y_index] += 0.01
                system.update_scan(map_x_index, map_y_index, 0.01)
            else:
                print("ID, cell_x, cell_y, height_z", i, cell_x, cell_y, height_z)
                print("map_x_index_const, map_y_index_const", i, map_x_index, map_y_index)

        # energy consumption of MD and HD
        system.battery_consume()

        # execute action
        for MD in MD_set:
            action[str(MD.ID)], _, _ = ctrl[MD.ID].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                                                                   state=obs[str(MD.ID)]["state"],
                                                                   target_pos=MD.xyz_temp)
        for HD in HD_set:
            action[str(HD.ID+num_MD)], _, _ = ctrl[HD.ID+num_MD].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                                                                   state=obs[str(HD.ID+num_MD)]["state"],
                                                                   target_pos=HD.xyz_temp)
        # for i in range(ARGS.num_drones):
        #     action[str(i)], _, _ = ctrl[i].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
        #                                                            state=obs[str(i)]["state"],
        #                                                            target_pos=TARG_XYZS[i])
            # print(i)
            # print("TARG_XYZS", TARG_XYZS[i])
            # print("drone pos:", obs[str(i)]["state"][0:3])
            # action[str(i)], _, _ = ctrl[i].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
            #                                                        state=obs[str(i)]["state"],
            #                                                        target_pos=TARG_XYZS,
            #                                                        # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
            #                                                        target_rpy=INIT_RPYS[i, :]
            #                                                        )
            # action[str(i)], _, _ = ctrl[i].computeControlFromState(control_timestep = CTRL_EVERY_N_STEPS * env.TIMESTEP,
            #                                                        state = obs[str(i)]["state"],
            #                                                        target_pos = TARG_XYZS[i])
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

    # Game End
    # system.print_MDs()
    # system.print_HDs()




