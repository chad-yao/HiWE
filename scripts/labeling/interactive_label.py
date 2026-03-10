"""Interactive labeling for key intervals (press k for start, j for end)."""

import os
import h5py


def interactive_labeling_episodes(base_path, dataset_type="act"):
    """Interactively label key interval start/end in HDF5 episodes."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg")

    if dataset_type == "act":
        indices = [0, 1, 2, 40, 41]
        picture_key = "images/mid"
    else:
        indices = list(range(12)) + list(range(40, 48))  # 0-11, 40-47
        picture_key = "image/agent"

    for file_index in indices:
        file_path = os.path.join(base_path, f"episode_{file_index}.hdf5")
        if not os.path.exists(file_path):
            continue

        with h5py.File(file_path, "r+") as file:
            images = file[f"/observations/{picture_key}"][:]
            if "label" not in file:
                labels = file.create_dataset("label", (len(images),), dtype="i")
                labels[:] = 0
            else:
                labels = file["label"]

            current_index = 0

            def show_image(index):
                plt.clf()
                plt.imshow(images[index])
                plt.title(
                    f"Image {index + 1}/{len(images)} - Label: {labels[index]} "
                    f"- episode:{file_index}"
                )
                plt.draw()

            def on_scroll(event):
                nonlocal current_index
                if event.button == "up" and current_index > 0:
                    current_index -= 1
                elif event.button == "down" and current_index < len(images) - 1:
                    current_index += 1
                show_image(current_index)

            def on_key(event):
                nonlocal current_index
                if event.key == "enter":
                    file.flush()
                    plt.close(fig)
                    return
                elif event.key == "k":
                    labels[current_index] = 5  # key interval start
                    show_image(current_index)
                elif event.key == "j":
                    labels[current_index] = 6  # key interval end
                    show_image(current_index)
                elif event.key == " ":
                    labels[current_index] = 0
                    show_image(current_index)

            fig, ax = plt.subplots()
            fig.canvas.mpl_connect("scroll_event", on_scroll)
            fig.canvas.mpl_connect("key_press_event", on_key)
            show_image(current_index)
            plt.show()


def convert_labels_to_intervals(path, dataset_type="act"):
    """Convert 5/6 (start/end) labels to 0/1 (non-key/key) intervals."""
    if dataset_type == "act":
        indices = [0, 1, 2, 40, 41]
    else:
        indices = list(range(18))

    for file_index in indices:
        file_path = os.path.join(path, f"episode_{file_index}.hdf5")
        if not os.path.exists(file_path):
            continue

        with h5py.File(file_path, "r+") as file:
            labels = file["label"]
            in_interval = False
            for i in range(len(labels)):
                if labels[i] == 5 and not in_interval:
                    labels[i] = 1
                    in_interval = True
                elif labels[i] == 6 and in_interval:
                    labels[i] = 1
                    in_interval = False
                elif labels[i] not in (5, 6) and not in_interval:
                    labels[i] = 0
                else:
                    labels[i] = 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Dataset directory")
    parser.add_argument(
        "--dataset",
        choices=["act", "dp"],
        default="act",
        help="Dataset format",
    )
    parser.add_argument(
        "--mode",
        choices=["label", "convert"],
        default="label",
        help="label: interactive labeling | convert: 5/6 -> 0/1",
    )
    args = parser.parse_args()

    if args.mode == "label":
        interactive_labeling_episodes(args.path, args.dataset)
    else:
        convert_labels_to_intervals(args.path, args.dataset)
