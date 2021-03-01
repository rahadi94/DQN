import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
import numpy as np
from collections import deque
from Fleet_sim.location import closest_facility
from Fleet_sim.log import lg
import math
import pandas as pd
from math import ceil

A = 0.5
B = 0.1
C = 0.1
EPISODES = 20


def epsilon_decay(time):
    standardized_time = (time - A * EPISODES) / (B * EPISODES)
    cosh = np.cosh(math.exp(-standardized_time))
    epsilon = 1.1 - (1 / cosh + (time * C / EPISODES))
    return epsilon / 5


class Agent:
    def __init__(self):

        # Initialize atributes
        # self.env = env
        self._state_size = 7
        self._action_size = 5
        self._optimizer = Adam(learning_rate=0.001)
        self.batch_size = 32
        self.expirience_replay = deque(maxlen=20000)
        # self.episode = episode
        # Initialize discount and exploration rate
        self.gamma = 0.9
        self.Gamma = 0.9


        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

    def get_state(self, vehicle, charging_stations, vehicles, waiting_list, env):
        charging_station = closest_facility(charging_stations, vehicle)

        SOC = int((vehicle.charge_state - vehicle.charge_state % 10) / 10)
        if isinstance(SOC, np.ndarray):
            SOC = SOC[0]
        for j in range(0, 24):
            if j * 60 <= env.now % 1440 <= (j + 1) * 60:
                hour = j
        position = vehicle.position.id
        supply = len([v for v in vehicles if v.location.distance_1(vehicle.location) <= 4 and v.charge_state >= 30 and
                      vehicle.mode in ['idle', 'parking', 'circling', 'queue']])
        if isinstance(supply, np.ndarray):
            supply = supply[0]
        wl = len([t for t in waiting_list if t.origin.distance_1(vehicle.location) <= 5])
        if isinstance(wl, np.ndarray):
            wl = wl[0]
        queue = len(charging_station.plugs.queue)
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

        return np.array([SOC, hour, position, supply, queue, free_CS, wl])

    def store(self, state, action, reward, next_state, period):
        self.expirience_replay.append((state, action, reward, next_state, period))

    def _build_compile_model(self):
        '''if self.episode > 0:
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights("model.h5")
            # evaluate loaded model on test data
            model.compile(loss='mse', optimizer=self._optimizer)'''
        model = Sequential()
        # model.add(Embedding(self._state_size, 10, input_length=1))
        # model.add(Reshape((None,7)))
        model.add(Dense(512, activation='relu', input_dim=7))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))

        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state, episode):
        epsilon = epsilon_decay(episode)
        if np.random.rand() <= epsilon:
            if state[0, 0] >= 7:
                if state[0, 5] >= 1:
                    action = np.random.choice([0, 1, 2, 3, 4])
                else:
                    action = np.random.choice([0, 2, 4])
            else:
                if state[0, 5] >= 1:
                    action = np.random.choice([0, 1, 2, 4])
                else:
                    action = np.random.choice([0, 2, 4])
        else:
            q_values = self.q_network.predict(state)
            df = pd.DataFrame(q_values)
            if state[0, 0] > 70:
                if state[0, 5] >= 1:
                    action = np.argmax(df)
                else:
                    action = np.argmax(df.loc[0, [0, 2, 4]])
                    if action == 1:
                        action = 2
                    elif action == 2:
                        action = 4
            else:
                if state[0, 5] >= 1:
                    action = np.argmax(df.loc[0, [0, 1, 2, 4]])
                    if action == 3:
                        action = 4
                else:
                    action = np.argmax(df.loc[0, [0, 2, 4]])
                    if action == 1:
                        action = 2
                    elif action == 2:
                        action = 4
        return action

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)

        for state, action, reward, next_state, period in minibatch:

            target = self.q_network.predict(state)

            t = self.target_network.predict(next_state)
            df = pd.DataFrame(t)
            if state[0, 0] >= 7:
                if state[0, 5] >= 1:
                    df = df.loc[0, [0, 1, 2, 3, 4]]
                else:
                    df = df.loc[0, [0, 2, 4]]
            else:
                if state[0, 5] >= 1:
                    df = df.loc[0, [0, 1, 2, 4]]
                else:
                    df = df.loc[0, [0, 2, 4]]
            k = ceil((period) / 15)
            target[0][action] = reward + (self.gamma) ** k * np.amax(df)

            self.q_network.fit(state, target, epochs=1, verbose=0)

    def take_action(self, vehicle, charging_stations, vehicles, waiting_list, env, episode):
        state = self.get_state(vehicle, charging_stations, vehicles, waiting_list, env)
        state = state.reshape((1, 7))
        lg.info(f'new_state={state}, {vehicle.charging_count}')
        action = self.act(state, episode)
        vehicle.old_location = vehicle.location
        lg.info(f'new_action={action}, new_state={state}, {vehicle.charging_count}')
        vehicle.r = float(-(vehicle.reward['charging'] + vehicle.reward['distance'] * 0.80 - vehicle.reward[
            'revenue'] - vehicle.reward['discharging'] + vehicle.reward['queue'] / 30 + vehicle.reward['parking']
                            / 120 + vehicle.reward['missed']))
        reward = vehicle.r
        vehicle.final_reward += vehicle.r
        if vehicle.old_state is not None:
            period = env.now - vehicle.old_time
            self.store(vehicle.old_state, vehicle.old_action, reward, state, period)
        if len(self.expirience_replay) > self.batch_size:
            if len(self.expirience_replay) % 10 == 1:
                self.retrain(self.batch_size)
        if len(self.expirience_replay) % 50 == 1:
            self.alighn_target_model()
        vehicle.old_time = env.now
        vehicle.old_state = state
        vehicle.old_action = action
        vehicle.reward['revenue'] = 0
        vehicle.reward['distance'] = 0
        vehicle.reward['charging'] = 0
        vehicle.reward['queue'] = 0
        vehicle.reward['parking'] = 0
        vehicle.reward['missed'] = 0
        vehicle.reward['discharging'] = 0
        return action
