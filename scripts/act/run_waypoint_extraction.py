"""Extract waypoints from ACT dataset using hierarchical key intervals."""

import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm

from hiwe.waypoint import dp_waypoint_selection


def get_label_sequences_from_hdf5(dataset_dir, num_episodes):
    """Load actions, qpos, and label sequences from ACT dataset."""
    file_paths = [
        os.path.join(dataset_dir, f"episode_{i}.hdf5")
        for i in range(num_episodes)
    ]
    actions, qposes, all_label_sequences = [], [], []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        with h5py.File(file_path, "r") as f:
            actions.append(f["/action"][:])
            qposes.append(f["/observations/qpos"][:])
            labels = f["/label-test"][:]
            if labels.size > 0:
                current_label = labels[0]
                start_index = 0
                seq = []
                for i in range(1, len(labels)):
                    if labels[i] != current_label:
                        seq.append([current_label, [start_index, i - 1]])
                        current_label = labels[i]
                        start_index = i
                seq.append(
                    [current_label, [start_index, len(labels) - 1]]
                )
                all_label_sequences.append(seq)
            else:
                all_label_sequences.append([])

    return actions, qposes, all_label_sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="ACT dataset dir")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument(
        "--threshold",
        type=float,
        nargs=2,
        default=[0.012, 0.004],
        help="[non-key, key] error thresholds",
    )
    parser.add_argument(
        "--waypoint_key",
        default="/waypoints_ssl_hwe",
        help="HDF5 key for saving waypoints",
    )
    parser.add_argument("--plot_3d", action="store_true")
    args = parser.parse_args()

    actions, qposes, labels_sequences = get_label_sequences_from_hdf5(
        args.dataset, args.num_episodes
    )
    threshold = args.threshold

    for episode_, (eef_qpos, labels) in enumerate(
        zip(qposes, labels_sequences)
    ):
        waypoints = []
        for label, indice in labels:
            adjusted_start, adjusted_end = indice[0], indice[1]
            if adjusted_end <= adjusted_start + 1:
                continue
            sub_waypoints = dp_waypoint_selection(
                env=None,
                actions=eef_qpos[adjusted_start:adjusted_end],
                gt_states=eef_qpos[adjusted_start:adjusted_end],
                err_threshold=threshold[label],
                pos_only=True,
            )
            waypoints += [adjusted_start + wp for wp in sub_waypoints]
            waypoints += [adjusted_start, adjusted_end]

        waypoints = sorted(set(waypoints))

        file_path = os.path.join(args.dataset, f"episode_{episode_}.hdf5")
        with h5py.File(file_path, "r+") as root:
            eef_qpos_data = root["action"][()]
            print(
                f"Episode {episode_}: {len(eef_qpos_data)} frames -> "
                f"{len(waypoints)} waypoints"
            )
            if args.waypoint_key in root:
                del root[args.waypoint_key]
            root[args.waypoint_key] = waypoints

    print("Done.")


if __name__ == "__main__":
    main()
