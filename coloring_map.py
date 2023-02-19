'''
data from https://gadm.org/download_country_v3.html#google_vignette

'''


import geopandas as gpd
import matplotlib.pyplot as plt

import csp

reg_group_name = 'NAME_1'  # 'NAME_2' 'NAME_1'

# Load country map data
if reg_group_name == 'NAME_1':
    country = gpd.read_file('./data/gadm36_LTU_shp/gadm36_LTU_1.shp')
else:
    country = gpd.read_file('./data/gadm36_LTU_shp/gadm36_LTU_2.shp')


# Group regions by a common attribute (e.g. region name or id)
regions = country.groupby(reg_group_name)
#regions = country.groupby('NAME_2')



# Create an empty dictionary to hold the neighboring regions for each region
neighbor_dict = {}

# Iterate over the regions and find the neighboring regions
for name, group in regions:
    neighbors = []
    for _, other in regions:
        if ((name != other[reg_group_name].iloc[0]) & (group.geometry.touches(other.geometry.iloc[0]))).any():
            neighbors.append(other[reg_group_name].iloc[0])
    neighbor_dict[name] = neighbors

# Print the dictionary of neighboring regions
print(neighbor_dict)


def dict_to_string(neighbor_dict):
    output = ""
    for key in neighbor_dict:
        neighbors = " ".join(neighbor_dict[key])
        output += f"{key}: {neighbors}; "
    return output[:-2]


output_string = dict_to_string(neighbor_dict)
print(output_string)


regions_csp = csp.MapColoringCSP(list('RGBY'), output_string )  # RGB   RGBY

solution = csp.backtracking_search(regions_csp, select_unassigned_variable=csp.mrv, order_domain_values=csp.lcv, inference=csp.mac)
print(solution)


# Assign unique colors to each region
colors = plt.cm.get_cmap('tab20', len(regions))  # choose a colormap and number of colors
color_dict = {name: colors(i) for i, name in enumerate(regions.groups)}

if solution != None:
    colors = plt.cm.get_cmap('tab20', 4)
    for region, color in solution.items():
        print(f"{region}: {color}")
        if color == 'R':
            color_dict[region] = colors(0)
        if color == 'G':
            color_dict[region] = colors(1)
        if color == 'B':
            color_dict[region] = colors(2)
        if color == 'Y':
            color_dict[region] = colors(3)







# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Plot each region on the same axis
for name, group in regions:
    group.plot(ax=ax, color=color_dict[name])



# Calculate the centroid of each region and add the region name as a label
for name, group in regions:
    #group = group.buffer(0.01)  # apply a small buffer to fix any gaps or overlaps
    centroid = group.centroid.values[0]
    plt.text(centroid.x, centroid.y, name, fontsize=12, ha='center', va='center')

# Set the title of the map
plt.title('Map of Regions')

# Display the map
plt.show()




print()