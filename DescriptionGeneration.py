import pandas as pd
import numpy as np

meta_file = 'ca_meta.csv'
adj_file = 'ca_rn_adj.npy'
output_file = 'station_descriptions.csv'

station_meta = pd.read_csv(meta_file)
adj_matrix = np.load(adj_file)

direction_map = {'N': 'North', 'S': 'South', 'E': 'East', 'W': 'West'}

def generation_description(idx, row, adj_matrix, id_list, threshold = 0.2):
    node_id = row['ID']
    county = row['County']
    fwy = row['Fwy']
    direction = direction_map.get(str(row['Direction']).strip(), row['Direction'])
    lanes = row['Lanes']
    lat, lng = row['Lat'], row['Lng']
    sensor_type = row['Type']
    conn_texts = []
    for j, strength in enumerate(adj_matrix[idx]):
        if strength > threshold and idx != j:
            neighbor_id = id_list[j]
            conn_texts.append(f"{neighbor_id} (dependency score: {strength:.2f})")
    conn_text = ", ".join(conn_texts) if conn_texts else "None"

    # Compose description
    description = (
        f"Sensor ID: {node_id}. "
        f"It is located in {county} County, California, on highway {fwy}, heading {direction}. "
        f"The road segment has {lanes} lanes and is monitored by a {sensor_type} sensor. "
        f"The geographic coordinates are ({lat:.4f}, {lng:.4f}). "
        f"Strongly connected sensors: {conn_text}."
    )
    return description

id_list = station_meta['ID']
descriptions = []

for idx, row in station_meta.iterrows():
    desc = generation_description(idx, row, adj_matrix, id_list)
    descriptions.append(desc)

data_dict = {'ID': id_list, 'description': descriptions}
df = pd.DataFrame(data_dict)
df.to_csv(output_file, index = False, encoding='utf-8-sig')