from docplex.mp.model import Model

from Fleet_sim.log import lg


def available_vehicle(vehicles, trip, SOC_threshold=20, max_distance=10):
    available_vehicles = list()
    for vehicle in vehicles:
        if vehicle.mode in ['idle', 'parking', 'circling', 'queue']:
            distance_to_pickup = vehicle.location.distance_1(trip.origin)
            if distance_to_pickup <= max_distance:
                distance_to_dropoff = trip.distance
                charge_consumption = (distance_to_pickup + distance_to_dropoff) * \
                                     vehicle.fuel_consumption * 100.0 / vehicle.battery_capacity
                if charge_consumption + SOC_threshold <= vehicle.charge_state:
                    available_vehicles.append(vehicle)
    return available_vehicles


def matching(vehicles, trips):
    try:
        mdl = Model("CS_development")
        vehicle_range = []
        for i in vehicles:
            vehicle_range.append(i.id)
        trip_range = []
        for j in trips:
            trip_range.append(j.id)

        d = {}
        for i in vehicles:
            for j in trips:
                d[i.id, j.id] = i.location.distance_1(j.origin)
                if i.mode in ['charging', 'discharging']:
                    d[i.id, j.id] += 2
                if isinstance(d[i.id, j.id], list):
                    d[i.id, j.id] = float([i.id, j.id][0])

        l = {}
        for j in trips:
            l[j.id] = j.distance
            if isinstance(l[j.id], list):
                l[j.id] = float(l[j.id][0])

        p = {}
        for j in trips:
            p[j.id] = j.revenue
            if isinstance(p[j.id], list):
                p[j.id] = float(p[j.id][0])

        SOC = {}
        for i in vehicles:
            SOC[i.id] = i.charge_state
            if isinstance(SOC[i.id], list):
                SOC[i.id] = float([i.id][0])

        x = mdl.binary_var_matrix(vehicle_range, trip_range, name='x')

        for i in vehicles:
            mdl.add_constraint(mdl.sum(x[i.id, j.id] for j in trips) <= 1, 'C1')
        for j in trips:
            mdl.add_constraint(mdl.sum(x[i.id, j.id] for i in vehicles) <= 1, 'C2')
        for j in trips:
            for i in vehicles:
                mdl.add_constraint(x[i.id, j.id] * (d[i.id, j.id] + l[j.id]) * 0.4 <= SOC[i.id] - 20, 'C2')

        mdl.maximize(mdl.sum(x[i.id, j.id] * (p[j.id] - d[i.id, j.id] * 0.01) for i in vehicles for j in trips))

        mdl.solve()
        #mdl.report()
        pairs = []
        for i in vehicles:
            for j in trips:
                if x[i.id, j.id].solution_value != 0:
                    pairs.append(dict(vehicle=i, trip=j))
        lg.info(f'NoT = {len(trips)}, NoV = {len(vehicles)}, NoM = {len(pairs)}')

    except:
        lg.info('Solving the model fails')
        pairs = []
        for trip in trips:
            available_vehicles = available_vehicle(vehicles, trip)
            distances = [vehicle.location.distance_1(trip.origin) for vehicle in available_vehicles]
            # If there is no available vehicle, add the trip to the waiting list
            if len(available_vehicles) >= 1:
                vehicle = [x for x in available_vehicles
                           if x.location.distance_1(trip.origin) == min(distances)][0]
                vehicles.remove(vehicle)
                pairs.append(dict(vehicle=vehicle, trip=trip))
        lg.info(f'NoT = {len(trips)}, NoV = {len(vehicles)}, NoM = {len(pairs)}')
    return pairs

