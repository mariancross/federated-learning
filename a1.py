from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def main():
    input_file = Path("resources/fmidata.csv")
    input_data = pd.read_csv(input_file)
    stations = list(input_data["station"].unique())

    central_train_data = list()
    central_valid_data = list()

    acc_val_error = 0

    fl_net = nx.Graph()
    for station in stations:
        station_data = input_data[input_data["station"] == station].sort_values(by="day")
        valid_day = station_data.iloc[-1]["day"]

        train_data = station_data[station_data["day"] < valid_day][["tmin", "tmax"]].to_numpy()
        valid_data = station_data[station_data["day"] == valid_day][["tmin", "tmax"]].to_numpy()

        fl_net.add_node(station, train_data=train_data, valid_data=valid_data)

        model = LinearRegression(fit_intercept=True)
        model.fit(train_data[:, 0:1], train_data[:, 1])
        pred = model.predict(valid_data[:, 0:1])

        mse = np.mean((valid_data[:, 1] - pred) ** 2)
        acc_val_error += mse

        central_train_data.extend(train_data)
        central_valid_data.extend(valid_data)

    acc_val_error /= len(fl_net)
    print(f"mean error {acc_val_error}")

    central_train_data = np.array(central_train_data)
    central_valid_data = np.array(central_valid_data)

    model = LinearRegression(fit_intercept=True)
    model.fit(central_train_data[:, 0:1], central_train_data[:, 1])

    pred = model.predict(central_train_data[:, 0:1])
    mse = np.mean((central_train_data[:, 1] - pred) ** 2)
    print(f"Train error {mse}")

    pred = model.predict(central_valid_data[:, 0:1])
    mse = np.mean((central_valid_data[:, 1] - pred) ** 2)
    print(f"Valid error {mse}")


if __name__ == '__main__':
    main()
