import osmnx as ox

G = ox.io.load_graphml('./mynetwork.graphml')
'''bbox = [52.10, 52.80, 13.00, 13.80]
G = ox.graph_from_bbox(bbox[1], bbox[0], bbox[3], bbox[2], network_type='drive', retain_all=False,
                       truncate_by_edge=True, simplify=False)'''

'''G = ox.graph_from_place('Berlin, Germany', network_type='drive', buffer_dist=5000, retain_all=False)'''
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
