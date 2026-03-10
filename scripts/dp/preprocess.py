"""Convert DP (robomimic) dataset to ACT-style per-episode HDF5 format."""

import os
import h5py
import numpy as np


def create_individual_hdf5_files(input_file_path, output_directory):
    """Split DP dataset into per-episode HDF5 files."""
    os.makedirs(output_directory, exist_ok=True)

    with h5py.File(input_file_path, "r") as input_file:
        demo_keys = [
            k for k in input_file["/data"].keys() if k.startswith("demo_")
        ]

        for demo_key in demo_keys:
            demo_index = demo_key.split("_")[1]
            output_file_path = os.path.join(
                output_directory, f"episode_{demo_index}.hdf5"
            )

            with h5py.File(output_file_path, "w") as output_file:
                input_dataset_path = f"/data/{demo_key}/obs/agentview_image"
                output_dataset_path = "observations/image/agent"
                data = input_file[input_dataset_path][()]
                output_file.create_dataset(output_dataset_path, data=data)
                print(f"Created {output_file_path}")


def find_max_image_count(output_directory):
    """Find max frame count across all episodes."""
    max_image_count = 0
    for file_name in sorted(os.listdir(output_directory)):
        if file_name.endswith(".hdf5"):
            file_path = os.path.join(output_directory, file_name)
            with h5py.File(file_path, "r") as file:
                count = file["observations/image/agent"].shape[0]
                max_image_count = max(max_image_count, count)
    return max_image_count


def expand_image_sequences(output_directory, max_image_count):
    """Pad shorter episodes to max length by repeating last frame."""
    for file_name in sorted(os.listdir(output_directory)):
        if file_name.endswith(".hdf5"):
            file_path = os.path.join(output_directory, file_name)
            with h5py.File(file_path, "a") as file:
                data = file["observations/image/agent"][()]
                current_length = data.shape[0]

                if current_length < max_image_count:
                    last_frame = data[-1]
                    padding = np.tile(
                        last_frame,
                        (max_image_count - current_length, 1, 1, 1),
                    )
                    data = np.vstack((data, padding))
                    del file["observations/image/agent"]
                    file.create_dataset(
                        "observations/image/agent", data=data
                    )
                    print(f"Expanded {file_path} to {data.shape[0]} frames")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input DP HDF5")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    create_individual_hdf5_files(args.input, args.output)
    max_count = find_max_image_count(args.output)
    expand_image_sequences(args.output, max_count)
