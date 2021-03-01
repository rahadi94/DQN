import pandas as pd
import numpy as np
import simpy
import random
#from Fleet_sim.DQN import Agent
from Fleet_sim.location import find_zone, closest_facility
from Fleet_sim.log import lg
from Fleet_sim.read import charging_cost
from Fleet_sim.trip import Trip
from Fleet_sim.Matching import matching
from math import ceil


class Model:

    def __init__(self, env, vehicles, charging_stations, zones, parkings, simulation_time, episode, learner):
        self.t = []
        self.episode = episode
        self.parkings = parkings
        self.zones = zones
        self.charging_stations = charging_stations
        self.vehicles = vehicles
        self.trip_list = []
        self.waiting_list = []
        self.simulation_time = simulation_time
        self.env = env
        self.trip_start = env.event()
        self.demand_generated = []
        self.discharging_demand_generated = []
        self.utilization = []
        self.vehicle_id = None
        self.learner = learner

    def parking_task(self, vehicle):
        # the selected parking lot is the closest free one
        if vehicle.mode in ['circling', 'idle']:
            free_PL = [x for x in self.parkings if len(x.capacity.queue) == 0]
            if len(free_PL) >= 1:
                parking = closest_facility(free_PL, vehicle)
                with parking.capacity.request() as req:
                    yield req
                    yield self.env.process(self.park(vehicle, parking))
            else:
                return

    def park(self, vehicle, parking):
        # all vehicles must start from parking
        if self.env.now <= 5:
            vehicle.parking(parking)
            yield vehicle.parking_stop
        else:
            # each vehicle cruise 10 mins before sending to parking
            if vehicle.mode == 'idle':
                vehicle.mode = 'circling'
                lg.info(f'vehicle {vehicle.id} starts cruising at {self.env.now}')
                circling_interruption = vehicle.circling_stop
                vehicle.t_start_circling = self.env.now
                circling_finish = self.env.timeout(10.5)
                parking_events = yield circling_interruption | circling_finish

                # vehicle sends to the request if there is any
                if circling_interruption in parking_events:
                    vehicle.charge_state -= vehicle.SOC_consumption(10 * (15 / 60))
                    lg.info(f'vehicle {vehicle.id} interrupts cruising at {self.env.now}')

                # vehicle sends to the parking if there is no request
                elif circling_finish in parking_events:
                    if vehicle.mode == 'circling':
                        lg.info(f'vehicle {vehicle.id} stops cruising at {self.env.now}')
                        circling_time = max(float(self.env.now - vehicle.t_start_circling), 0)
                        vehicle.charge_state -= vehicle.SOC_consumption(circling_time * (15 / 60))
                        vehicle.send_parking(parking)
                        yield self.env.timeout(vehicle.time_to_parking)
                        t_start_parking = self.env.now
                        vehicle.parking(parking)

                        # vehicle stays in parking until it assigns to a request or send for charging/relocating
                        yield vehicle.parking_stop
                        vehicle.reward['parking'] += (self.env.now - t_start_parking)
                        k = ceil((self.env.now - vehicle.decision_time)/15)
                        vehicle.reward['parking'] = vehicle.reward['parking'] * self.learner.Gamma ** k

    def relocate_check(self, vehicle):
        # we need first update the state of all zones
        for zone in self.zones:
            zone.update(self.vehicles)
        vehicle.position = find_zone(vehicle.location, self.zones)
        time = self.env.now

        # this checks whether there is surplus supply or not
        for i in range(0, 24):
            if i * 60 <= time % 1440 <= (i + 1) * 60:
                hour = i
        if vehicle.charge_state >= 50 and vehicle.mode in ['idle', 'parking'] and len(
                vehicle.position.list_of_vehicles) >= vehicle.position.demand.iloc[0, hour]:
            return True

    def relocate_task(self, vehicle):
        # first we find the zones with the lack of supply
        time = self.env.now
        for i in range(0, 24):
            if i * 60 <= time % 1440 <= (i + 1) * 60:
                hour = i
        target_zones = [z for z in self.zones if len(z.list_of_vehicles) <= z.demand.iloc[0, hour]]

        # then we select the closest zones from those ones
        if len(target_zones) > 1:
            target_zone = closest_facility(target_zones, vehicle)
            if vehicle.mode == 'parking':
                vehicle.parking_stop.succeed()
                vehicle.parking_stop = self.env.event()
            self.env.process(self.relocate(vehicle, target_zone))

    def relocate(self, vehicle, target_zone):
        vehicle.relocate(target_zone)
        target_zone.update(self.vehicles)
        yield self.env.timeout(vehicle.time_to_relocate)
        vehicle.finish_relocating(target_zone)
        vehicle.relocating_end.succeed()
        vehicle.relocating_end = self.env.event()

    def charge_check(self, vehicle):
        # we choose charging decision using a q-learning algorithm
        if self.env.now > 10 and vehicle.mode in ['idle', 'parking']:
            # updating starts from the second decision for each vehicle
            '''if vehicle.charging_count > 0:
                self.learner.update_value(vehicle, self.charging_stations, self.vehicles, self.waiting_list)'''
            vehicle.decision_time = round(self.env.now)
            action = self.learner.take_action(vehicle, self.charging_stations, self.vehicles, self.waiting_list,
                                              self.env, self.episode)
            vehicle.action = action
            vehicle.charging_count += 1
            return action

    def charge_task(self, vehicle, action):
        # finding the charging station based on the charging decision
        # action 0 ==> closest CS
        # action 1 ==> closest free CS
        # action 2 ==> closest fast CS
        if action == 0:
            charging_station = closest_facility(self.charging_stations, vehicle)
        elif action == 1:
                    free_CS = [x for x in self.charging_stations if x.plugs.count < x.capacity]
                    if len(free_CS) >= 1:
                        charging_station = closest_facility(free_CS, vehicle)
                    # we send the vehicle to the closest CS, if there is no free one
                    else:
                        charging_station = closest_facility(self.charging_stations, vehicle)
        elif action == 2:
            fast_CS = [x for x in self.charging_stations if x.power == 50 / 60]
            charging_station = closest_facility(fast_CS, vehicle)

        # vehicle sends to the CS and enters the queue using a priority coming from its SOC
        yield self.env.process(self.start_charge(charging_station, vehicle))
        prio = int((vehicle.charge_state - vehicle.charge_state % 10) / 10)
        if isinstance(prio, np.ndarray):
            prio = prio[0]
        req = charging_station.plugs.request(priority=prio)
        vehicle.mode = 'queue'
        # the vehicle either starts charging after the queue or interrupt the queue if it matches with a request
        events = yield req | vehicle.queue_interruption
        # start charging if it does not assigns to a request
        if req in events:
            vehicle.charging_demand['time_start'] = self.env.now
            lg.info(f'Vehicle {vehicle.id} starts charging at {self.env.now}')
            vehicle.t_start_charging = self.env.now
            vehicle.reward['queue'] += (vehicle.t_start_charging - vehicle.t_arriving_CS)
            k = ceil((self.env.now - vehicle.decision_time)/15)
            vehicle.reward['queue'] = vehicle.reward['queue'] * self.learner.Gamma ** k
            if isinstance(vehicle.reward['queue'], np.ndarray):
                vehicle.reward['queue'] = vehicle.reward['queue'][0]
            vehicle.mode = 'charging'
            charging = self.env.process(self.finish_charge(charging_station, vehicle))
            # even while charging it can matches with a request (but with more costs)
            yield charging | vehicle.charging_interruption
            charging_station.plugs.release(req)
            req.cancel()

            # if it interrupts before finishing the charging event, we need to update everything
            if not charging.triggered:
                charging.interrupt()
                lg.info(f'Vehicle {vehicle.id} stops charging at {self.env.now}')
                return

        # if it interrupts the queue before the charging is being started, we need to update everything
        else:
            vehicle.reward['queue'] += (self.env.now - vehicle.t_arriving_CS)
            k = ceil((self.env.now - vehicle.decision_time)/15)
            vehicle.reward['queue'] = vehicle.reward['queue'] * self.learner.Gamma ** k
            if isinstance(vehicle.reward['queue'], np.ndarray):
                vehicle.reward['queue'] = vehicle.reward['queue'][0]
            vehicle.charging_demand['SOC_end'] = vehicle.charge_state
            lg.info(f'vehicle {vehicle.id} interrupts the queue')
            req.cancel()
            charging_station.plugs.release(req)
            return

    def start_charge(self, charging_station, vehicle):
        vehicle.send_charge(charging_station)
        # we need this information for CS planning problem
        vehicle.charging_demand = dict(vehicle_id=vehicle.id, time_send=self.env.now,
                                       time_enter=self.env.now + vehicle.time_to_CS,
                                       time_start=None, SOC_end=None,
                                       SOC_send=vehicle.charge_state, lat=vehicle.location.lat,
                                       long=vehicle.location.long,
                                       v_hex=vehicle.position.hexagon,
                                       CS_location=[charging_station.location.lat, charging_station.location.long],
                                       v_position=vehicle.position.id, CS_position=charging_station.id,
                                       distance=vehicle.location.distance(charging_station.location)[0])
        yield self.env.timeout(vehicle.time_to_CS)
        vehicle.charging(charging_station)
        vehicle.t_arriving_CS = self.env.now

    def finish_charge(self, charging_station, vehicle):
        # the vehicle either finishes the charging process or interrupts it
        try:
            yield self.env.timeout(vehicle.charge_duration)
            vehicle.finish_charging(charging_station)
            vehicle.charging_demand['SOC_end'] = vehicle.charge_state
            vehicle.charging_end.succeed()
            vehicle.charging_end = self.env.event()
        except simpy.Interrupt:
            old_SOC = vehicle.charge_state
            vehicle.charge_state += float((charging_station.power * (float(self.env.now) - vehicle.t_start_charging)) \
                                          / (vehicle.battery_capacity / 100))
            vehicle.charging_demand['SOC_end'] = vehicle.charge_state
            for j in range(0, 24):
                if j * 60 <= self.env.now % 1440 <= (j + 1) * 60:
                    h = j
            vehicle.reward['charging'] += (vehicle.charge_state - old_SOC) / 100 * 50 * charging_cost[h] / 100
            vehicle.profit -= (vehicle.charge_state - old_SOC) / 100 * 50 * charging_cost[h] / 100
            k = ceil((self.env.now - vehicle.decision_time)/15)
            vehicle.reward['charging'] = vehicle.reward['charging'] * self.learner.Gamma ** k
            if isinstance(vehicle.reward['charging'], np.ndarray):
                vehicle.reward['charging'] = vehicle.reward['charging'][0]
            vehicle.costs['charging'] += (vehicle.charging_threshold - vehicle.charge_state) * \
                                         charging_cost[h] / 100
            lg.info(f'Warning!!!Charging state of vehicle {vehicle.id} is {vehicle.charge_state} at {self.env.now} ')
        self.demand_generated.append(vehicle.charging_demand)

    def discharge_task(self, vehicle):
        # vehicle sends for discharging only if it has more than 70% charge and there is free CS
        if vehicle.charge_state > 70:
            free_CS = [x for x in self.charging_stations if x.plugs.count < x.capacity]
            if len(free_CS) >= 1:
                charging_station = closest_facility(free_CS, vehicle)
            else:
                charging_station = closest_facility(self.charging_stations, vehicle)
            yield self.env.process(self.start_discharge(charging_station, vehicle))
            # it enters the station with a priority
            req = charging_station.plugs.request(priority=5)
            vehicle.mode = 'queue'
            # the vehicle either starts discharging after the queue or interrupt the queue if it matches with a request
            events = yield req | vehicle.queue_interruption
            if req in events:
                vehicle.discharging_demand['time_start'] = self.env.now
                lg.info(f'Vehicle {vehicle.id} starts discharging at {self.env.now}')
                vehicle.t_start_discharging = self.env.now
                vehicle.mode = 'discharging'
                discharging = self.env.process(self.finish_discharge(vehicle=vehicle,
                                                                     charging_station=charging_station))
                # even while charging it can matches with a request (but with more costs)
                yield discharging | vehicle.discharging_interruption
                charging_station.plugs.release(req)
                req.cancel()
                vehicle.discharging_demand['SOC_end'] = vehicle.charge_state

                # if it interrupts before finishing the charging event, we need to update everything
                if not discharging.triggered:
                    discharging.interrupt()
                    lg.info(f'Vehicle {vehicle.id} stops discharging at {self.env.now}')
                    return
            else:
                lg.info(f'vehicle {vehicle.id} interrupts the queue')
                req.cancel()
                charging_station.plugs.release(req)
                return
        else:
            return

    def start_discharge(self, charging_station, vehicle):
        vehicle.send_charge(charging_station)
        lg.info(f'vehicle {vehicle.id} is sent for discharging')
        # we need this information for CS planning problem
        vehicle.discharging_demand = dict(vehicle_id=vehicle.id, time_send=self.env.now,
                                          time_enter=self.env.now + vehicle.time_to_CS,
                                          time_start=None, SOC_end=None,
                                          SOC_send=vehicle.charge_state, lat=vehicle.location.lat,
                                          long=vehicle.location.long,
                                          v_hex=vehicle.position.hexagon,
                                          CS_location=[charging_station.location.lat, charging_station.location.long],
                                          v_position=vehicle.position.id, CS_position=charging_station.id,
                                          distance=vehicle.location.distance(charging_station.location)[0])
        yield self.env.timeout(vehicle.time_to_CS)
        vehicle.discharging(charging_station)
        vehicle.t_arriving_CS = self.env.now

    def finish_discharge(self, charging_station, vehicle):
        # the vehicle either finishes the charging process or interrupts it
        try:
            yield self.env.timeout(vehicle.discharge_duration)
            vehicle.finish_discharging(charging_station)
            vehicle.discharging_demand['SOC_end'] = vehicle.charge_state
            vehicle.discharging_end.succeed()
            vehicle.discharging_end = self.env.event()
        except simpy.Interrupt:
            old_SOC = vehicle.charge_state
            vehicle.charge_state -= float((charging_station.power * (float(self.env.now) - vehicle.t_start_discharging))
                                          / (vehicle.battery_capacity / 100))
            vehicle.discharging_demand['SOC_end'] = vehicle.charge_state
            for j in range(0, 24):
                if j * 60 <= self.env.now % 1440 <= (j + 1) * 60:
                    h = j
            vehicle.reward['discharging'] += (vehicle.charge_state - old_SOC) / 100 * 50 * charging_cost[h] / 100
            vehicle.profit += (vehicle.charge_state - old_SOC) / 100 * 50 * charging_cost[h] / 100
            k = ceil((self.env.now - vehicle.decision_time) / 15)
            vehicle.reward['discharging'] = vehicle.reward['discharging'] * self.learner.Gamma ** k
            if isinstance(vehicle.reward['discharging'], np.ndarray):
                vehicle.reward['discharging'] = vehicle.reward['charging'][0]
            lg.info(f'Warning!!!Charging state of vehicle {vehicle.id} is {vehicle.charge_state} at {self.env.now} ')
        self.discharging_demand_generated.append(vehicle.discharging_demand)

    def trip_cancellation(self, trip, vehicle):
        # the request cancels after matching considering the distance with the matched vehicle
        pass

    def pick_trip(self, trip, vehicle):
        vehicle.send(trip)
        trip.mode = 'assigned'
        self.trip_cancellation(trip, vehicle)
        try:
            yield self.env.timeout(vehicle.time_to_pickup)
            self.waiting_list.remove(trip)
            vehicle.pick_up(trip)
            trip.mode = 'in vehicle'
        except simpy.Interrupt:
            trip.mode = 'cancelled'
            self.trip_list.append(trip)
            vehicle.mode = 'idle'
            vehicle.trip_cancellation.succeed()
            vehicle.trip_cancellation = self.env.event()

    def take_trip(self, trip, vehicle):
        trip_pick = self.env.process(self.pick_trip(trip, vehicle))
        yield trip_pick | trip.cancellation

        if not trip_pick.triggered:
            trip_pick.interrupt()
            lg.info(f'Trip {trip.id} cancelled at {self.env.now}')
        else:
            yield self.env.timeout(trip.duration)
            vehicle.drop_off(trip)
            vehicle.trip_end.succeed()
            vehicle.trip_end = self.env.event()
            self.vehicle_id = vehicle.id
            trip.mode = 'finished'
            self.trip_list.append(trip)
            trip.info['mode'] = 'finished'
            vehicle.reward['revenue'] += max(((trip.distance * 1.11 + trip.duration * 0.31) + 2), 5) - \
                                         float(trip.info['waiting_time']) * 0.10
            vehicle.profit += max(((trip.distance * 1.11 + trip.duration * 0.31) + 2), 5) - \
                                         float(trip.info['waiting_time']) * 0.10
            k = ceil((self.env.now - vehicle.decision_time)/15)
            vehicle.reward['revenue'] = vehicle.reward['revenue'] * self.learner.Gamma ** k
            if isinstance(vehicle.reward['revenue'], np.ndarray):
                vehicle.reward['revenue'] = vehicle.reward['revenue'][0]

    def trip_task(self):
        vehicles = list()
        for vehicle in self.vehicles:
            if vehicle.mode in ['idle', 'parking', 'circling', 'queue']:
                vehicles.append(vehicle)
            if vehicle.mode in ['charging']:
                power = 11 / 60
                try:
                    duration = self.env.now - vehicle.t_start_charging
                except:
                    duration = 0
                soc = vehicle.charge_state + ((power * duration) / (vehicle.battery_capacity / 100))
                if soc >= 40 and duration >= 10:
                    vehicles.append(vehicle)
            if vehicle.mode in ['discharging']:
                try:
                    duration = self.env.now - vehicle.t_start_discharging
                except:
                    duration = 0
                if duration >= 10:
                    vehicles.append(vehicle)
        trips = [x for x in self.waiting_list if x.mode == 'unassigned']
        pairs = matching(vehicles, trips)
        if len(pairs) == 0:
            return
        for i in pairs:
            vehicle = i['vehicle']
            trip = i['trip']
            if vehicle.mode == 'parking':
                vehicle.parking_stop.succeed()
                vehicle.parking_stop = self.env.event()
            elif vehicle.mode == 'circling':
                vehicle.circling_stop.succeed()
                vehicle.circling_stop = self.env.event()
            elif vehicle.mode == 'queue':
                vehicle.queue_interruption.succeed()
                vehicle.queue_interruption = self.env.event()
            elif vehicle.mode == 'charging':
                vehicle.charging_interruption.succeed()
                vehicle.charging_interruption = self.env.event()
            elif vehicle.mode == 'discharging':
                vehicle.discharging_interruption.succeed()
                vehicle.discharging_interruption = self.env.event()
            self.env.process(self.take_trip(trip, vehicle))

            # yield self.env.timeout(0.001)

    def trip_generation(self, zone):
        j = 0
        while True:
            j += 1
            trip = Trip(self.env, (j, zone.id), zone)
            yield self.env.timeout(trip.interarrival)
            self.trip_start.succeed()
            self.trip_start = self.env.event()
            self.trip = trip
            trip.info['arrival_time'] = self.env.now
            self.waiting_list.append(trip)
            lg.info(f'Trip {trip.id} is received at {self.env.now}')
            trip.start_time = self.env.now

    def missed_trip(self):
        while True:
            for trip in self.waiting_list:
                if trip.mode == 'unassigned' and self.env.now > (trip.start_time + 3):
                    r = random.uniform(0, 1)
                    if r < 0.1:
                        trip.mode = 'missed'
                        trip.info['mode'] = 'missed'
                        self.trip_list.append(trip)
                        self.waiting_list.remove(trip)
                        lg.info(f'trip {trip.id} is missed at {self.env.now}')
                elif trip.mode == 'unassigned' and self.env.now > (trip.start_time + 5):
                    r = random.uniform(0, 1)
                    if r < 0.5:
                        trip.mode = 'missed'
                        trip.info['mode'] = 'missed'
                        self.trip_list.append(trip)
                        self.waiting_list.remove(trip)
                        lg.info(f'trip {trip.id} is missed at {self.env.now}')
                elif trip.mode == 'unassigned' and self.env.now > (trip.start_time + 10):
                    trip.mode = 'missed'
                    trip.info['mode'] = 'missed'
                    self.trip_list.append(trip)
                    self.waiting_list.remove(trip)
                    lg.info(f'trip {trip.id} is missed at {self.env.now}')
                if trip.mode == 'missed':
                    vehicle_responsible = [x for x in self.vehicles if x.mode in
                                           ['charging', 'discharging', 'queue', 'ertc'] and
                                           x.old_location.distance_1(trip.origin) <= 10
                                           and x.charge_state > 30]
                    vehicle_responsible_0 = [x for x in self.vehicles if x.location.distance_1(trip.origin) <= 10 and
                                           x.mode in ['charging', 'discharging', 'queue', 'ertc']
                                           and x.charge_state > 30]
                    vehicle_responsible_1 = ([x for x in self.vehicles if x.location.distance_1(trip.origin) <= 10
                                              and x.mode in ['idle', 'parking'] and x.charge_state <= 25
                                              and x.charging_count > 0])
                    for vehicle in vehicle_responsible:
                        vehicle.reward['missed'] += 50
                        k = ceil((self.env.now - vehicle.decision_time) / 15)
                        vehicle.reward['missed'] = vehicle.reward['missed'] * self.learner.Gamma ** k
                    for vehicle in vehicle_responsible_0:
                        vehicle.reward['missed'] += 50
                        k = ceil((self.env.now - vehicle.decision_time) / 15)
                        vehicle.reward['missed'] = vehicle.reward['missed'] * self.learner.Gamma ** k
                    for vehicle in vehicle_responsible_1:
                        vehicle.reward['missed'] += 50
                        k = ceil((self.env.now - vehicle.decision_time) / 15)
                        vehicle.reward['missed'] = vehicle.reward['missed'] * self.learner.Gamma ** k
            yield self.env.timeout(1)

    def hourly_charging_relocating(self):
        while True:
            yield self.env.timeout(60)
            for vehicle in self.vehicles:
                if vehicle.mode == 'parking':
                    action = self.charge_check(vehicle)
                    if action in [0, 1, 2]:
                        self.env.process(self.charge_task(vehicle, action))
                        yield self.env.timeout(0.001)
                    elif action == 3:
                        self.env.process(self.discharge_task(vehicle))
                        yield self.env.timeout(0.001)
                    elif self.relocate_check(vehicle):
                        self.relocate_task(vehicle)
                        yield self.env.timeout(0.001)

    def run(self):
        while True:
            yield self.env.timeout(2)
            if len(self.waiting_list) >= 1:
                self.trip_task()

    def run_vehicle(self, vehicle):
        while True:
            if self.env.now == 0:
                self.env.process(self.parking_task(vehicle))

            event_trip_end = vehicle.trip_end
            event_charging_end = vehicle.charging_end
            event_trip_cancellation = vehicle.trip_cancellation
            event_relocating_end = vehicle.relocating_end
            event_discharging_end = vehicle.discharging_end
            events = yield event_trip_end | event_charging_end \
                           | event_relocating_end | event_discharging_end | event_trip_cancellation

            if event_trip_end in events:
                lg.info(f'A vehicle gets idle at {self.env.now}')
                action = self.charge_check(vehicle)
                if action in [0, 1, 2]:
                    self.env.process(self.charge_task(vehicle, action))
                    yield self.env.timeout(0.001)
                elif action == 3:
                    self.env.process(self.discharge_task(vehicle))
                    yield self.env.timeout(0.001)
                else:
                    if self.relocate_check(vehicle):
                        self.relocate_task(vehicle)
                        yield self.env.timeout(0.001)
                    else:
                        self.env.process(self.parking_task(vehicle))
                        yield self.env.timeout(0.001)

            if event_charging_end in events:
                lg.info(f'A vehicle gets charged at {self.env.now}')
                if self.relocate_check(vehicle):
                    self.relocate_task(vehicle)
                    yield self.env.timeout(0.001)
                else:
                    self.env.process(self.parking_task(vehicle))
                    yield self.env.timeout(0.001)

            if event_discharging_end in events:
                lg.info(f'A vehicle gets discharged at {self.env.now}')
                if self.relocate_check(vehicle):
                    self.relocate_task(vehicle)
                    yield self.env.timeout(0.001)
                else:
                    self.env.process(self.parking_task(vehicle))
                    yield self.env.timeout(0.001)

            if event_trip_cancellation in events:
                lg.info(f'Trip gets canceled at {self.env.now}')
                self.env.process(self.parking_task(vehicle))
                yield self.env.timeout(0.001)

            if event_relocating_end in events:
                lg.info(f'vehicle {vehicle.id} finishes relocating at {self.env.now}')
                self.env.process(self.parking_task(vehicle))
                yield self.env.timeout(0.001)

    def obs_Ve(self, vehicle):
        while True:
            t_now = self.env.now
            self.t.append(t_now)
            vehicle.info['SOC'].append(vehicle.charge_state)
            vehicle.info['location'].append([vehicle.location.lat, vehicle.location.long])
            vehicle.info['position'].append(vehicle.position)
            vehicle.info['mode'].append(vehicle.mode)
            yield self.env.timeout(1)

    def obs_CS(self, charging_station):
        while True:
            charging_station.queue.append([charging_station.plugs.count, len(charging_station.plugs.queue)])
            yield self.env.timeout(1)

    def obs_PK(self, parking):
        while True:
            parking.queue.append(parking.capacity.count)
            yield self.env.timeout(1)

    def save_results(self, episode):
        trips_info = []
        for i in self.trip_list:
            trips_info.append(i.info)
        results = pd.DataFrame(trips_info)
        results_charging_demand = pd.DataFrame(self.demand_generated)
        results_discharging_demand = pd.DataFrame(self.discharging_demand_generated)
        model_json = self.learner.q_network.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.learner.q_network.save_weights("model.h5")
        # with pd.ExcelWriter("results.xlsx", engine="openpyxl", mode='a') as writer:
        results.to_csv(f'results/trips{episode}.csv')
        results_charging_demand.to_csv(f'results/charging{episode}.csv')
        results_discharging_demand.to_csv(f'results/discharging{episode}.csv')

        pd_ve = pd.DataFrame()
        pd_reward = pd.DataFrame()
        for j in self.vehicles:
            pd_ve = pd_ve.append(pd.DataFrame([j.info["mode"], j.info['SOC'], j.info['location']]))
            pd_reward = pd_reward.append(pd.DataFrame([j.total_rewards['state'], j.total_rewards['action'],
                                                       j.total_rewards['reward']]))
        pd_ve.to_csv(f'results/vehicles{episode}.csv')
        pd_reward.to_csv(f'results/rewards{episode}.csv')

        pd_cs = pd.DataFrame()
        for c in self.charging_stations:
            pd_cs = pd_cs.append([c.queue])
        pd_cs.to_csv(f'results/CSs{episode}.csv')

        """for p in self.parkings:
            pd.DataFrame([p.queue]).to_excel(writer, sheet_name='PK_%s' % p.id)"""
