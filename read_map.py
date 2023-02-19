import os
import csv
'''
write function that scans this list and selects max and min values of 'lat' and 'lng' atributes for each distinct value in 'country' attribute and puts this info in dictionary with key country
'''
# Get the absolute path of the directory containing the program
dir_path = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the data directory
data_path = os.path.join(dir_path, 'data')

# Construct the full path to the file
file_path = os.path.join(data_path, 'worldcities.csv')

# Open the file in read mode with UTF-8 encoding
with open(file_path, 'r', encoding='utf-8') as file:
    # Create a csv reader object
    reader = csv.DictReader(file)
    # Read the header row
    headers = next(reader)
    # Create a list to store the data
    data = []
    # Loop over the first 10 rows
    for i, row in enumerate(reader):
        # Append each row as a dictionary to the list
        data.append(dict(row))
        # Break the loop after reading 10 rows
        #if i == 9:
            #break


'''
This function takes a list of dictionaries as input (data) and returns a dictionary with the country as the key and a dictionary of the minimum and maximum latitude and longitude values as the value. The function iterates over the rows in the list, retrieves the country, latitude, and longitude values, and stores them in a separate dictionary. If a country hasn't been seen yet, it is added to the dictionary with its latitude and longitude values as the minimum and maximum. If the country has already been seen, the minimum and maximum latitude and longitude values are updated based on the current row's values.
'''
def extract_lat_lng_min_max(data):
    country_lat_lng = {}
    for row in data:
        country = row['country']
        lat = float(row['lat'])
        lng = float(row['lng'])
        if country not in country_lat_lng:
            country_lat_lng[country] = {'lat_min': lat, 'lat_max': lat, 'lng_min': lng, 'lng_max': lng}
        else:
            country_lat_lng[country]['lat_min'] = min(country_lat_lng[country]['lat_min'], lat)
            country_lat_lng[country]['lat_max'] = max(country_lat_lng[country]['lat_max'], lat)
            country_lat_lng[country]['lng_min'] = min(country_lat_lng[country]['lng_min'], lng)
            country_lat_lng[country]['lng_max'] = max(country_lat_lng[country]['lng_max'], lng)
    return country_lat_lng


country_lat_lng_dic = extract_lat_lng_min_max(data)


#country_lat_lng_dic = {k: v for k, v in country_lat_lng_dic_tmp.items() if v['lat_min'] != v['lat_max'] and v['lng_min'] != v['lng_max']}




# Print the data list to check if it's correct
'''
for i, row in enumerate(data):
    print(row)
    if i == 9:
        break
'''



'''
This function takes a latitude, longitude, a dictionary of country latitude and longitude minimum and maximum values (country_lat_lng_dic), and a resolution (defaults to 1600x900). 
It first finds the country that the GPS coordinate belongs to. Then, it calculates the corresponding pixel position by transforming the GPS coordinate into a value between 0 and 1 
for both the x and y axes. The x value is calculated by dividing the difference between the longitude and the minimum longitude by the difference between the maximum and minimum longitudes,
 and then multiplying by the x resolution. The y value is calculated by dividing the difference between the maximum latitude and the latitude by the difference 
 between the maximum and minimum latitudes, and then multiplying by the y resolution. Finally, the x and y values are returned as a tuple of integers.

'''
def lat_lng_to_pixel(lat, lng, country_lat_lng_dic, country, resolution=(1600, 900)):  # resolution=(1600, 900)
    #country = None
    #for key in country_lat_lng_dic:
    #    if lat >= country_lat_lng_dic[key]['lat_min'] and lat <= country_lat_lng_dic[key]['lat_max'] and \
    #       lng >= country_lat_lng_dic[key]['lng_min'] and lng <= country_lat_lng_dic[key]['lng_max']:
    #        country = key
    #        break

    #if country is None:
    #    #raise ValueError("The GPS coordinate does not belong to any country in the country_lat_lng_dic.")
    #    x = 0
    #    y = 0
    #else:

    lat_min = country_lat_lng_dic[country]['lat_min']
    lat_max = country_lat_lng_dic[country]['lat_max']
    lng_min = country_lat_lng_dic[country]['lng_min']
    lng_max = country_lat_lng_dic[country]['lng_max']

    if lng_max != lng_min and lat_max != lat_min:
        x = (lng - lng_min) / (lng_max - lng_min) * resolution[0]
        #y = (lat_max - lat) / (lat_max - lat_min) * resolution[1]
        y = (lat - lat_min ) / (lat_max - lat_min) * resolution[1]
    else:
        x = 0.5 * resolution[0]
        y = 0.5 * resolution[1]


    return (int(x), int(y) )

#print(country_lat_lng_dic['Lithuania'])

#aa = lat_lng_to_pixel(54.6833, 25.2833, country_lat_lng_dic,'Lithuania')

#print(aa)

'''
This function uses a list comprehension to iterate through the input data and checks each dictionary's country attribute. If the attribute is equal to the country parameter, the dictionary is added to the result list. The function returns the result list.
'''
#def filter_by_country(data, country):
#    return [item for item in data if item.get('country') == country]


def filter_by_country(data, country):
    result = []
    counter = 0
    for item in data:
        if item.get('country') == country:
            result.append(item)
            counter += 1
            if counter >= 60:
                break
    return result


for record in data:
    city = record['city']
    lat = float(record['lat'])
    lng = float(record['lng'])
    country = record['country']
    x, y = lat_lng_to_pixel(lat, lng, country_lat_lng_dic, country)
    record['x'] = x
    record['y'] = y



country_data = filter_by_country(data, 'Lithuania')

#print(country_data)










import random

def generate_graph(country_data):
    # Create a dictionary to store the graph
    graph = {}

    # Loop through each dictionary in the country_data list
    for record in country_data:
        city = record['city']
        x = record['x']
        y = record['y']

        # Add the city as a node in the graph, with x and y as coordinates
        graph[city] = (x, y)

    # Loop through each city in the graph
    for city1 in graph:
        # Generate a random number of edges for this city
        num_edges = random.randint(2, 5)

        # Find the closest `num_edges` cities to city1
        closest_cities = sorted(graph.keys(), key=lambda c: abs(graph[c][0] - graph[city1][0]) + abs(graph[c][1] - graph[city1][1]))[1:num_edges + 1]

        # Connect city1 to each of the closest cities
        for city2 in closest_cities:
            graph[city1] = (graph[city1][0], graph[city1][1], city2)

    return graph


import matplotlib.pyplot as plt

def plot_graph(graph):
    x_coords = [graph[city][0] for city in graph]
    y_coords = [graph[city][1] for city in graph]

    for city in graph:
        x1 = graph[city][0]
        y1 = graph[city][1]

        for connected_city in graph[city][2:]:
            x2 = graph[connected_city][0]
            y2 = graph[connected_city][1]

            plt.plot([x1, x2], [y1, y2], 'b-')

    plt.scatter(x_coords, y_coords)
    for city in graph:
        plt.annotate(city, (graph[city][0], graph[city][1]))
    plt.show()


'''
country_data = [{'city': 'New York', 'lat': 40.7128, 'lng': -74.0060},
                {'city': 'London', 'lat': 51.5074, 'lng': 0.1278},
                {'city': 'Paris', 'lat': 48.8566, 'lng': 2.3522}]
'''
graph = generate_graph(country_data)
#plot_graph(graph)


'''
In this example, the country_data list is processed to extract the x and y values and the city names, which are stored in the points and nodes lists, respectively. The Delaunay function is used to generate the Delaunay triangulation of the points, and the edges of the graph are extracted from the simplices of the triangulation. Each edge is represented as a tuple of two node names. The function returns the list of edges.
'''

from scipy.spatial import Delaunay

def generate_graph_2(country_data):
    points = []
    nodes = []
    for data in country_data:
        points.append([data['x'], data['y']])
        nodes.append(data['city'])
    dt = Delaunay(points)
    edges = []
    for simplex in dt.simplices:
        edges.append((nodes[simplex[0]], nodes[simplex[1]]))
        edges.append((nodes[simplex[1]], nodes[simplex[2]]))
        edges.append((nodes[simplex[2]], nodes[simplex[0]]))
    return edges

'''
In this updated version, a label for each node is added to the plot using the plt.annotate function. The label is the city name corresponding to the node, and the position of the label is the x and y value of the node. The rest of the function remains unchanged.
'''
import matplotlib.pyplot as plt
def plot_graph_2(country_data, edges):
    points = []
    nodes = []
    for data in country_data:
        points.append([data['x'], data['y']])
        nodes.append(data['city'])
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.scatter(x, y)
    for edge in edges:
        node1 = nodes.index(edge[0])
        node2 = nodes.index(edge[1])
        x1, y1 = points[node1]
        x2, y2 = points[node2]
        plt.plot([x1, x2], [y1, y2], 'k-')
    for i, node in enumerate(nodes):
        plt.annotate(node, (x[i], y[i]))
    plt.show()


edges = generate_graph_2(country_data)

plot_graph_2(country_data , edges)






import math

# This change should resolve the TypeError and allow the get_edges_with_distance function to run successfully.
def get_edges_with_distance(country_data, edges):
    nodes = {}
    for data in country_data:
        nodes[data['city']] = (float(data['lat']), float(data['lng']))
    new_edges = []
    for edge in edges:
        node1 = nodes[edge[0]]
        node2 = nodes[edge[1]]
        lat1, lng1 = node1
        lat2, lng2 = node2
        dist = haversine(lat1, lng1, lat2, lng2)
        dist = int(dist)
        if dist < 1000:
            new_edges.append((edge[0], edge[1], dist))
    return new_edges



def haversine(lat1, lng1, lat2, lng2):
    # Convert latitude and longitude to spherical coordinates in radians.
    degrees_to_radians = math.pi / 180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1) * degrees_to_radians
    phi2 = (90.0 - lat2) * degrees_to_radians

    # theta = longitude
    theta1 = lng1 * degrees_to_radians
    theta2 = lng2 * degrees_to_radians

    # Compute spherical distance from spherical coordinates.
    cos = (math.sin(phi1) * math.sin(phi2) * math.cos(theta1 - theta2) +
           math.cos(phi1) * math.cos(phi2))
    arc = math.acos(cos)

    # Multiply arc by the radius of the earth to get length.
    earth_radius = 6371  # kilometers
    return arc * earth_radius




new_edges = get_edges_with_distance(country_data, edges)


#print(new_edges)



def get_romania_map_string(new_edges):
    romania_map = dict()
    for edge in new_edges:
        node1, node2, distance = edge
        if node1 not in romania_map:
            romania_map[node1] = dict()
        romania_map[node1][node2] = distance

    romania_map_str = 'maps.romania_map = UndirectedGraph(dict(\n'
    for node, edges in romania_map.items():
        node = node.replace(" ", "_").replace("-", "_").replace("’", "_")
        #dest = dest.replace(" ", "_")
        #node_str = f"    {node}=dict({', '.join([f'{dest}={dist}' for dest, dist in edges.items()])})"

        node_str = "    " + node + "=dict("
        for dest, dist in edges.items():
            dest = dest.replace(" ", "_").replace("-", "_").replace("’", "_")
            node_str += f"{dest}={dist}, "
        #node_str = node_str[:-2] + ")"
        node_str = node_str[:-1] + ")"

        node_str = node_str[:-2] + ")"
        romania_map_str += node_str + ',\n'
    romania_map_str = romania_map_str[:-2] + '\n))'
    return romania_map_str




print()
print(get_romania_map_string(new_edges))
print()

def get_romania_locations_string(country_data):
    locations_string = "maps.romania_map.locations = dict(\n"
    for location in country_data:
        node = location['city'].replace(" ", "_").replace("-", "_").replace("’", "_")
        locations_string += f"    {node}=({location['x']},{location['y']}),\n"
    locations_string = locations_string[:-2] + "\n)"
    return locations_string


print()
print(get_romania_locations_string(country_data))
print()

filename = "d:/all_maps.py"
with open(filename, "w", encoding="utf-8") as file:
    file.write("\nfrom search import *\n\n")

for country in country_lat_lng_dic.keys():
    country_data = filter_by_country(data, country)
    if len(country_data) < 4:
        continue
    print(f"Data for {country}: {country_data}")
    edges = generate_graph_2(country_data)
    new_edges = get_edges_with_distance(country_data, edges)
    romania_map_str = get_romania_map_string(new_edges)
    locations_string = get_romania_locations_string(country_data)

    strings = [f"# {country} with number of cities: {len(country_data)}\n", romania_map_str, "\n", locations_string + "\n\n\n"]
    with open(filename, "a", encoding="utf-8") as file:
        for string in strings:
            file.write(string + "\n")






