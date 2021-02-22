import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
import numpy as np
from collections import deque
from Fleet_sim.location import closest_facility
from Fleet_sim.log import lg


class Agent:
    def __init__(self, env, episode):

        # Initialize atributes
        self.env = env
        self._state_size = 7
        self._action_size = 3
        self._optimizer = Adam(learning_rate=0.01)
        self.batch_size = 32
        self.expirience_replay = deque(maxlen=2000)
        self.episode = episode
        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.Gamma = 0.9
        self.epsilon = 0.1

        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

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

        return np.array([SOC, hour, position, supply, queue, free_CS, wl])

    def store(self, state, action, reward, next_state, period):
        self.expirience_replay.append((state, action, reward, next_state, period))

    def _build_compile_model(self):
        if self.episode > 0:
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights("model.h5")
            print("Loaded model from disk")
            # evaluate loaded model on test data
            model.compile(loss='mse', optimizer=self._optimizer)
        model = Sequential()
        #model.add(Embedding(self._state_size, 10, input_length=1))
        #model.add(Reshape((None,7)))
        model.add(Dense(50, activation='relu', input_dim=7))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))

        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([0, 1, 2])

        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)

        for state, action, reward, next_state, period in minibatch:

            target = self.q_network.predict(state)

            t = self.target_network.predict(next_state)
            target[0][action] = reward + (self.gamma)**period * np.amax(t)

            self.q_network.fit(state, target, epochs=1, verbose=0)

    def take_action(self, vehicle, charging_stations, vehicles, waiting_list):
        state = self.get_state(vehicle, charging_stations, vehicles, waiting_list)
        state = state.reshape((1, 7))
        lg.info(f'new_state={state}, {vehicle.charging_count}')
        action = self.act(state)
        vehicle.old_location = vehicle.location
        lg.info(f'new_action={action}, new_state={state}, {vehicle.charging_count}')
        vehicle.r = float(-(vehicle.reward['charging'] + vehicle.reward['distance'] * 0.80 - vehicle.reward[
            'revenue'] - vehicle.reward['discharging'] + vehicle.reward['queue'] / 30 + vehicle.reward['parking']
                            / 120 + vehicle.reward['missed']))
        reward = vehicle.r
        vehicle.final_reward += vehicle.r
        if vehicle.old_state is not None:
            period = self.env.now - vehicle.old_time
            self.store(vehicle.old_state, vehicle.old_action, reward, state, period)
        if len(self.expirience_replay) > self.batch_size:
            if len(self.expirience_replay) % 10 == 1:
                self.retrain(self.batch_size)
        if len(self.expirience_replay) % 50 == 1:
            self.alighn_target_model()
        vehicle.old_time = self.env.now
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
