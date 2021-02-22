import numpy as np
import pandas as pd
from Fleet_sim.location import closest_facility
from Fleet_sim.log import lg
from math import ceil
import math

A = 0.5
B = 0.1
C = 0.1
EPISODES = 20


def epsilon_decay(time):
    standardized_time = (time - A * EPISODES) / (B * EPISODES)
    cosh = np.cosh(math.exp(-standardized_time))
    epsilon = 1.1 - (1 / cosh + (time * C / EPISODES))
    return epsilon / 1.5


class RL_agent:
    def __init__(self, env, episode):
        self.env = env
        SOC = range(11)
        time = range(24)
        position = range(89)
        supply = range(4)
        queue = range(2)
        waiting_list = range(4)
        free_CS = range(2)
        self.episode = episode
        self.Gamma = 0.9
        if episode == 0:
            index = pd.MultiIndex.from_product([SOC, time, position, supply, queue, free_CS, waiting_list],
                                               names=['SOC', 'time', 'position', 'supply', 'queue', 'free_CS',
                                                      'waiting_list'])
            self.q_table = pd.DataFrame(-np.random.rand(len(index), 5), index=index)
            self.q_table.columns = ['0', '1', '2', '3', '4']
            self.q_table['counter_0'] = 0
            self.q_table['counter_1'] = 0
            self.q_table['counter_2'] = 0
            self.q_table['counter_3'] = 0
            self.q_table['counter_4'] = 0
            '''
            0 ==> charge in closest CS
            1 ==> charge in the closest free CS
            2 ==> charge in the closest fast CS
            3 ==> discharge in the closest free CS
            4 ==> Not charge neither discharge
            '''
        else:
            self.q_table = pd.read_csv('q_table.csv')
            self.q_table = self.q_table.set_index(['SOC', 'time', 'position', 'supply', 'queue', 'free_CS'
                                                      , 'waiting_list'])
        '''if episode == 1:
            self.q_table['counter'] = 0'''

    def get_state(self, vehicle, charging_stations, vehicles, waiting_list):
        charging_station = closest_facility(charging_stations, vehicle)

        SOC = int((vehicle.charge_state - vehicle.charge_state % 10) / 10)
        if isinstance(SOC, np.ndarray):
            SOC = SOC[0]
        for j in range(0, 24):
            if j * 60 <= self.env.now % 1440 <= (j + 1) * 60:
                hour = j
        position = vehicle.position.id
        supply = len([v for v in vehicles if v.location.distance_1(vehicle.location) <= 4 and v.charge_state >= 30 and
                      vehicle.mode in ['idle', 'parking', 'circling', 'queue']])
        if supply == 0:
            supply = 0
        elif supply < 5:
            supply = 1
        elif supply < 10:
            supply = 2
        else:
            supply = 3
        if isinstance(supply, np.ndarray):
            supply = supply[0]
        wl = len([t for t in waiting_list if t.origin.distance_1(vehicle.location) <= 5])
        if wl == 0:
            wl = 0
        elif wl < 5:
            wl = 1
        elif wl < 10:
            wl = 2
        else:
            wl = 3
        if isinstance(wl, np.ndarray):
            wl = wl[0]
        q = len(charging_station.plugs.queue)
        if q == 0:
            queue = 0
        else:
            queue = 1
        if isinstance(queue, np.ndarray):
            queue = queue[0]
        number_free_CS = 0
        for CS in charging_stations:
            if CS.plugs.count < CS.capacity:
                number_free_CS += 1
        if number_free_CS > 1:
            free_CS = 1
        else:
            free_CS = 0
        return (SOC, hour, position, supply, queue, free_CS, wl)

    def take_action(self, vehicle, charging_stations, vehicles, waiting_list):
        epsilon = epsilon_decay(self.episode)
        state = self.get_state(vehicle, charging_stations, vehicles, waiting_list)
        vehicle.old_location = vehicle.location
        if np.random.random() > epsilon:
            if vehicle.charge_state > 70:
                if state[5] >= 1:
                    action = np.argmax(self.q_table.loc[state, ['0', '1', '2', '3', '4']])
                else:
                    action = np.argmax(self.q_table.loc[state, ['0', '2', '4']])
                    if action == 1:
                        action = 2
                    elif action == 2:
                        action = 4

            else:
                if state[5] >= 1:
                    action = np.argmax(self.q_table.loc[state, ['0', '1', '2', '4']])
                    if action == 3:
                        action = 4
                else:
                    action = np.argmax(self.q_table.loc[state, ['0', '2', '4']])
                    if action == 1:
                        action = 2
                    elif action == 2:
                        action = 4
        else:
            if vehicle.charge_state > 70:
                if state[5] >= 1:
                    action = np.random.choice([0, 1, 2, 3, 4])
                else:
                    action = np.random.choice([0, 2, 4])
            else:
                if state[5] >= 1:
                    action = np.random.choice([0, 1, 2, 4])
                else:
                    action = np.random.choice([0, 2, 4])
        vehicle.old_state = state
        vehicle.old_action = action
        vehicle.old_time = self.env.now
        vehicle.reward['revenue'] = 0
        vehicle.reward['distance'] = 0
        vehicle.reward['charging'] = 0
        vehicle.reward['queue'] = 0
        vehicle.reward['parking'] = 0
        vehicle.reward['missed'] = 0
        vehicle.reward['discharging'] = 0
        lg.info(f'new_action={action}, new_state={state}, {vehicle.charging_count}')
        return action

    def update_value(self, vehicle, charging_stations, vehicles, waiting_list):
        self.q_table.loc[vehicle.old_state, f'counter_{vehicle.old_action}'] += 1
        a = self.q_table.loc[vehicle.old_state, f'counter_{vehicle.old_action}']
        alpha = 1 / a
        GAMMA = self.Gamma
        state = self.get_state(vehicle, charging_stations, vehicles, waiting_list)
        if vehicle.charge_state > 70:
            if state[5] >= 1:
                q = float(max(self.q_table.loc[state, ['0', '1', '2', '3', '4']]))
            else:
                q = float(max(self.q_table.loc[state, ['0', '2', '4']]))
        else:
            if state[5] >= 1:
                q = float(max(self.q_table.loc[state, ['0', '1', '2', '4']]))
            else:
                q = float(max(self.q_table.loc[state, ['0', '2', '4']]))

        vehicle.r = float(-(vehicle.reward['charging'] + vehicle.reward['distance'] * 0.80 - vehicle.reward[
            'revenue'] - vehicle.reward['discharging'] * 0.3 + vehicle.reward['queue'] / 30 + vehicle.reward['parking']
                            / 120 + vehicle.reward['missed']))
        vehicle.total_rewards['state'].append(vehicle.old_state)
        vehicle.total_rewards['action'].append(vehicle.old_action)
        vehicle.total_rewards['reward'].append(vehicle.r)
        vehicle.final_reward += vehicle.r

        # what if it changed meanwhile?
        vehicle.old_q = self.q_table.loc[vehicle.old_state, f'{vehicle.old_action}']
        k = ceil((self.env.now - vehicle.decision_time) / 15)

        self.q_table.loc[vehicle.old_state, f'{vehicle.old_action}'] = vehicle.old_q + \
                                                                       alpha * (vehicle.r + (
                GAMMA ** k) * q - vehicle.old_q)
        lg.info(f'old_action={vehicle.old_action}, old_state={vehicle.old_state}, new_state={state}, {vehicle.r}')
