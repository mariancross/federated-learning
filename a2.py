import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def create_fl_net():
    input_file = Path("resources/fmidata.csv")
    input_data = pd.read_csv(input_file)
    stations = list(input_data["station"].unique())

    fl_net = nx.Graph()

    for idx, name in enumerate(stations, start=1):
        station_data = input_data[input_data["station"] == name].sort_values(by="day")

        latitude = station_data["lat"].values[0]
        longitude = station_data["lon"].values[0]

        fl_net.add_node(idx, station=name, lat=latitude, lon=longitude)

        valid_day = station_data.iloc[-1]["day"]

        train_data = station_data[station_data["day"] < valid_day][["tmin", "tmax"]].sort_values(by="tmin").to_numpy()
        valid_data = station_data[station_data["day"] == valid_day][["tmin", "tmax"]].sort_values(by="tmin").to_numpy()

        train_data = np.hstack([np.ones((train_data.shape[0], 1)), train_data])
        valid_data = np.hstack([np.ones((valid_data.shape[0], 1)), valid_data])

        fl_net.nodes.get(idx)["trainset"] = train_data
        fl_net.nodes.get(idx)["valset"] = valid_data
        fl_net.nodes.get(idx)["model"] = LinearRegression(fit_intercept=False)

    distances = np.zeros((len(stations), len(stations)))

    for i in range(distances.shape[0]):
        node_i = fl_net.nodes.get(i + 1)
        for j in np.arange(i + 1, distances.shape[1], 1):
            node_j = fl_net.nodes.get(j + 1)
            distance = math.sqrt((node_i["lat"] - node_j["lat"]) ** 2 + (node_i["lon"] - node_j["lon"]) ** 2)
            distances[i, j] = distance
            distances[j, i] = distance

        nearest_neighbours = np.argsort(distances[i])[1:4]
        for neighbour in nearest_neighbours:
            fl_net.add_edge(i + 1, neighbour + 1)

    avg_node_degree = sum(deg for _, deg in fl_net.degree()) / fl_net.number_of_nodes()

    print(f"Number of nodes of the FL network: {fl_net.number_of_nodes()}")
    print(f"Is FL network a connected graph? {nx.is_connected(fl_net)}")
    print(f"Average node degree of the FL network: {avg_node_degree:.6f}")

    return fl_net, avg_node_degree

def gtv_min_block_coordinate(fl_net, alpha):
    nodes = fl_net.nodes()

    w = {}
    for node_idx in nodes:
        node = nodes.get(node_idx)

        x = node["trainset"][:, 0:2]
        y = node["trainset"][:, 2]

        model = node["model"]
        model.fit(x, y)
        w[node_idx] = model.coef_.copy()

    for t in range(100):
        w_new = {}
        for node_idx in nodes:
            node = nodes.get(node_idx)
            n = node["trainset"].shape[0]

            x_orig = node["trainset"][:, 0:2]
            y_orig = node["trainset"][:, 2]
            weights_orig = np.ones(n) / n

            x_aug = [x_orig]
            y_aug = [y_orig]
            weights_aug = [weights_orig]

            neighbours = fl_net.neighbors(node_idx)
            for neighbour in neighbours:
                x_pseudo = np.array([[1.0, 0.0], [0.0, 1.0]])
                y_pseudo = w[neighbour]
                weights_pseudo = np.array([alpha, alpha])

                x_aug.append(x_pseudo)
                y_aug.append(y_pseudo)
                weights_aug.append(weights_pseudo)

            x_aug = np.vstack(x_aug)
            y_aug = np.concatenate(y_aug)
            weights_aug = np.concatenate(weights_aug)

            model = node["model"]
            model.fit(x_aug, y_aug, sample_weight=weights_aug)
            w_new[node_idx] = model.coef_.copy()

        w = w_new
    return w


def apply_gtv_optim(fl_net, avg_node_degree):
    for alpha in [0.0, 1.0, 100.0]:
        w = gtv_min_block_coordinate(fl_net, alpha)

        numerator = sum(np.linalg.norm(w_i - w_j) ** 2 for w_i, w_j in fl_net.edges())
        denominator = sum(np.linalg.norm(w_i) ** 2 for w_i in w)

        var_model_params = (1 / avg_node_degree) * (numerator / denominator)
        print(f"alpha = {alpha}: Normalized variation of model parameters = {var_model_params:.6f}")

        n = fl_net.number_of_nodes()
        avg_val_error = 0.0
        for node_idx in fl_net.nodes():
            node = fl_net.nodes().get(node_idx)

            valset = node["valset"]
            x_val = valset[:, 0:2]
            y_val = valset[:, 2]

            model = node["model"]
            pred = model.predict(x_val)
            mse = np.mean((y_val - pred) ** 2)
            avg_val_error += (1 / valset.shape[0]) * mse

        avg_val_error /= n
        print(f"alpha = {alpha}: Average local validation error = {avg_val_error:.6f}")


def main():
    fl_net, avg_node_degree = create_fl_net()
    apply_gtv_optim(fl_net, avg_node_degree)


if __name__ == '__main__':
    main()
