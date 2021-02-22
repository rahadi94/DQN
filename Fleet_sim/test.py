'''/Users/ramin/PycharmProjects/Matching/venv/bin/python /Applications/PyCharm.app/Contents/plugins/python/helpers/pydev/pydevconsole.py --mode=client --port=54618
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/ramin/PycharmProjects/Matching'])
PyDev console: starting.
Python 3.7.0 (v3.7.0:1bf9cc5093, Jun 26 2018, 23:26:24)
[Clang 6.0 (clang-600.0.57)] on darwin
import random
random.uniform(0, 1)
0.8303970243159656
random.uniform(0, 1)
0.550814286606938
random.uniform(0, 1)
0.1388356873056804
random.uniform(0, 1)
0.5237341329302485
random.uniform(0, 1)
0.5525625455336556
random.uniform(0, 1)
0.48706244827547085
random.uniform(0, 1)
0.6667056190272237
random.uniform(0, 1)
... random.uniform(0, 1)
random.uniform(0, 1)
...
0.21392736838655724
from Fleet_sim.charging_station import ChargingStation
... from Fleet_sim.location import Location
... from Fleet_sim.model import Model, lg
... import simpy
... import random
... from Fleet_sim.parking import Parking
... from Fleet_sim.read import zones
... from Fleet_sim.vehicle import Vehicle
def generate_location():
...     return Location(random.uniform(52.40, 52.60), random.uniform(13.25, 13.55))
...
vehicles_data = []
... for i in range(10):
...     vehicle_data = dict(id=i, env=env, initial_location=generate_location(), capacity=50,
...                         charge_state=random.randint(70, 75), mode='idle')
...     vehicles_data.append(vehicle_data)
...
... vehicles = list()
...
... for data in vehicles_data:
...     vehicle = Vehicle(
...         data['id'],
...         data['env'],
...         data['initial_location'],
...         data['capacity'],
...         data['charge_state'],
...         data['mode']
...     )
...     vehicles.append(vehicle)
...
Traceback (most recent call last):
  File "<input>", line 3, in <module>
NameError: name 'env' is not defined
env = simpy.Environment()
...
vehicles_data = []
... for i in range(10):
...     vehicle_data = dict(id=i, env=env, initial_location=generate_location(), capacity=50,
...                         charge_state=random.randint(70, 75), mode='idle')
...     vehicles_data.append(vehicle_data)
...
... vehicles = list()
...
... for data in vehicles_data:
...     vehicle = Vehicle(
...         data['id'],
...         data['env'],
...         data['initial_location'],
...         data['capacity'],
...         data['charge_state'],
...         data['mode']
...     )
...     vehicles.append(vehicle)
...
from Fleet_sim.trip import Trip
trips = []
... j = 0
... # Trips are being generated randomly and cannot be rejected
... for j in range(50):
...     trip = Trip(self.env, (j, zone.id), zone)
...     trips.append(trip)
...
Traceback (most recent call last):
  File "<input>", line 5, in <module>
NameError: name 'self' is not defined
trips = []
... j = 0
... # Trips are being generated randomly and cannot be rejected
... for j in range(50):
...     trip = Trip(env, (j, zone.id), zone)
...     trips.append(trip)
...
Traceback (most recent call last):
  File "<input>", line 5, in <module>
NameError: name 'zone' is not defined
trips = []
... j = 0
... # Trips are being generated randomly and cannot be rejected
... for j in range(50):
...     trip = Trip(env, (j, id), zone)
...     trips.append(trip)
...
Traceback (most recent call last):
  File "<input>", line 5, in <module>
NameError: name 'zone' is not defined
import pandas as pd
...
... from Fleet_sim.Zone import Zone
...
...
... demand_table = pd.read_csv('demand_table.csv')
... OD_table = pd.read_csv('origin_destination.csv')
...
... z = 0
... zones = list()
... for hex in demand_table['h3_hexagon_id_start'].values:
...     z += 1
...     demand = (demand_table[demand_table['h3_hexagon_id_start'] == hex]).drop('h3_hexagon_id_start', axis=1)
...     destination = (OD_table[OD_table['h3_hexagon_id_start'] == hex]
...                    .drop('h3_hexagon_id_start', axis=1)).sort_values(by=z - 1, axis=1).T.reset_index()
...     zone = Zone(z, hex, demand, destination)
...     zones.append(zone)
... charging_threshold = [40, 45, 50, 55, 52, 50, 48, 45, 45, 42, 40, 40, 40, 40, 40, 38, 35, 32, 30, 30, 27, 30, 32, 35]
...
... charging_cost = [32.10, 32.00, 30.58, 30.02, 30.27, 33.19, 37.32, 44.45, 46.82, 45.70, 40.60,
...                      35.59, 34.08, 33.85, 34.45, 39.58, 45.84, 49.26, 52.40, 49.07, 43.55, 38.25, 38.10, 34.44]
...
trips = []
... j = 0
... # Trips are being generated randomly and cannot be rejected
... for j in range(50):
...     trip = Trip(env, (j, id), zone)
...     trips.append(trip)
...
trips = []
... for z in zones:
...     for j in range(50):
...         trip = Trip(env, (j, z.id), z)
...         trips.append(trip)
...
from Fleet_sim.Matching import matching
'''