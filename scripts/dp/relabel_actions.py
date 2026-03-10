"""Relabel actions in DP dataset using extracted waypoints."""

import numpy as np
import h5py


def relabel(input_file_path, output_directory=None):
    """Replace actions with waypoint-interpolated actions."""
    with h5py.File(input_file_path, "a") as input_file:
        demos = [
            k for k in input_file["data"].keys()
            if k.startswith("demo_")
        ]
        demos = sorted(demos, key=lambda x: int(x.split("_")[1]))

        for demo_key in demos:
            waypoint_path = f"/data/{demo_key}/waypoints_dp"
            if waypoint_path not in input_file:
                continue

            waypoints = input_file[waypoint_path][:]
            action_data = input_file[f"/data/{demo_key}/actions"][:]
            original_length = action_data.shape[0]
            action_new = []

            for i in range(original_length):
                nearest_index = original_length - 1
                for wp in waypoints:
                    if wp > i:
                        nearest_index = wp
                        break
                action_new.append(action_data[nearest_index])

            action_new = np.array(action_new[:original_length])
            action_path = f"/data/{demo_key}/action_new"
            if action_path in input_file:
                del input_file[action_path]
            input_file.create_dataset(action_path, data=action_new)
            print(f"Relabeled actions for {demo_key}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="DP HDF5 with waypoints")
    parser.add_argument("--episodes", default=None, help="Deprecated, ignored")
    args = parser.parse_args()

    relabel(args.input, args.episodes)
