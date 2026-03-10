"""Write predicted labels from episode HDF5s back to original DP dataset."""

import os
import h5py


def write_back(input_file_path, output_directory):
    """Copy label-test from per-episode files back to original DP HDF5."""
    with h5py.File(input_file_path, "a") as input_file:
        for file_name in sorted(os.listdir(output_directory)):
            if file_name.endswith(".hdf5"):
                demo_index = file_name.split("_")[1].split(".")[0]
                output_file_path = os.path.join(output_directory, file_name)

                with h5py.File(output_file_path, "r") as output_file:
                    original_count = input_file[
                        f"/data/demo_{demo_index}/obs/agentview_image"
                    ].shape[0]
                    label_test_data = output_file["label-test"][
                        :original_count
                    ]

                    label_path = f"/data/demo_{demo_index}/label-test"
                    if label_path in input_file:
                        del input_file[label_path]
                    input_file.create_dataset(
                        label_path, data=label_test_data
                    )
                    print(f"Wrote labels for demo_{demo_index}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original DP HDF5")
    parser.add_argument(
        "--episodes",
        required=True,
        help="Directory with episode_*.hdf5 files",
    )
    args = parser.parse_args()

    write_back(args.input, args.episodes)
