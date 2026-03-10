"""Train semi-supervised key interval segmentation model."""

import argparse
import copy
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from hiwe.segmentation.model import (
    KeyIntervalSegmenter,
    ImageSequenceHDF5Dataset,
    FocalLoss,
)


def temporal_loss_rnn(
    out, zcomp, w, labels, gamma=2.0, alpha=0.75, semi=False
):
    criterion = FocalLoss(gamma=gamma, alpha=alpha)
    sup_loss = torch.tensor(0.0, device=out.device)
    unsup_loss = torch.tensor(0.0, device=out.device)
    if semi and zcomp is not None:
        _, predicted_classes = torch.max(
            torch.softmax(
                zcomp.view(-1, zcomp.shape[-1]), dim=1
            ),
            dim=1,
        )
        unsup_loss = criterion(
            out.view(-1, out.shape[-1]), predicted_classes
        )
    if (labels >= 0).any():
        sup_loss = criterion(
            out.view(-1, out.shape[-1]), labels.view(-1)
        )
    return sup_loss + w * unsup_loss, sup_loss, unsup_loss


def weight_schedule(epoch, max_epochs, mult, max_w):
    if epoch == 0:
        return 0.0
    if epoch >= max_epochs:
        return max_w
    return max_w * np.exp(mult * (1.0 - float(epoch) / max_epochs) ** 2)


def calc_metrics(
    model, loader, gamma=2.0, alpha=0.75, device="cuda"
):
    model.eval()
    correct, total = 0, 0
    correct_1, total_1 = 0, 0
    correct_0, total_0 = 0, 0
    total_loss, total_count = 0.0, 0
    last_file_idx = None
    hidden = None

    with torch.no_grad():
        for images, labels, file_idx, start_idx in loader:
            images = images.to(device)
            labels = labels.to(device)
            if file_idx != last_file_idx:
                hidden = model.init_hidden(images.size(0))
                last_file_idx = file_idx
            outputs, hidden = model(images, hidden)
            hidden = hidden.detach()
            loss = temporal_loss_rnn(
                outputs, None, torch.tensor(0.0), labels,
                gamma=gamma, alpha=alpha, semi=False
            )[0]
            _, predicted = torch.max(outputs, dim=2)
            predicted = predicted.view(-1)
            labels_flat = labels.view(-1)
            total += labels_flat.size(0)
            correct += (predicted == labels_flat).sum().item()
            mask_1 = labels_flat == 1
            correct_1 += (predicted[mask_1] == labels_flat[mask_1]).sum().item()
            total_1 += mask_1.sum().item()
            mask_0 = labels_flat == 0
            correct_0 += (predicted[mask_0] == labels_flat[mask_0]).sum().item()
            total_0 += mask_0.sum().item()
            total_loss += loss.item()
            total_count += 1

    overall = 100 * correct / total if total else 0
    acc_1 = 100 * correct_1 / total_1 if total_1 else 0
    acc_0 = 100 * correct_0 / total_0 if total_0 else 0
    avg_acc = (acc_1 * (1 + alpha) + acc_0) / (2 + alpha)
    avg_loss = total_loss / total_count if total_count else 0
    return avg_loss, avg_acc, acc_1, acc_0


def predict_and_write_hdf5(
    hdf5_path,
    model,
    output_path=None,
    seq_len=None,
    size=112,
    image_key="/observations/image/agent",
    device="cuda",
):
    from torchvision.transforms.functional import to_tensor, resize

    model = model.to(device)
    model.eval()
    if output_path is None:
        output_path = hdf5_path

    with h5py.File(hdf5_path, "r") as file:
        images = file[image_key][()]
        total_frames = images.shape[0]
        seq_len = seq_len or total_frames
        num_sequences = total_frames // seq_len
        predictions = []

        for seq_idx in range(num_sequences):
            start_idx = seq_idx * seq_len
            end_idx = min(start_idx + seq_len, total_frames)
            mid_images = images[start_idx:end_idx]
            processed = []
            for img in mid_images:
                t = to_tensor(img)
                t = resize(t, (size, size), antialias=True)
                processed.append(t)
            x = torch.stack(processed).unsqueeze(0).to(device)
            with torch.no_grad():
                out, _ = model(x)
                _, pred = torch.max(out, dim=2)
                predictions.extend(pred.cpu().numpy().flatten().tolist())

    with h5py.File(output_path, "a") as file:
        if "label" in file:
            if "label-test" in file:
                del file["label-test"]
            file.create_dataset(
                "label-test", data=np.array(file["label"][()])
            )
        else:
            if "label-test" in file:
                del file["label-test"]
            file.create_dataset("label-test", data=np.array(predictions))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="HDF5 dataset root")
    parser.add_argument(
        "--format",
        choices=["act", "dp"],
        default="dp",
        help="act uses images/mid, dp uses image/agent",
    )
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.4)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--max_epochs_ramp", type=int, default=30)
    parser.add_argument("--ramp_mult", type=float, default=-15.0)
    parser.add_argument("--max_w", type=float, default=1.0)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--size", type=int, default=112)
    parser.add_argument("--labeled", type=str, default="0,1,2", help="Comma-separated indices")
    parser.add_argument("--train", type=str, default=None, help="Comma-separated train indices")
    parser.add_argument("--test", type=str, default=None, help="Comma-separated test indices")
    parser.add_argument("--predict_range", type=str, default="0-200", help="Range to predict after training")
    parser.add_argument("--output", default=None, help="Checkpoint/output dir")
    parser.add_argument("--semi", action="store_true", default=True)
    args = parser.parse_args()

    image_key = (
        "/observations/images/mid"
        if args.format == "act"
        else "/observations/image/agent"
    )

    labelset = [int(x) for x in args.labeled.split(",")]
    trainset = args.train
    if trainset:
        trainset = [int(x) for x in trainset.split(",")]
    else:
        trainset = list(
            set(range(200)) - set(range(40, 48))
        )  # default DP split
    testset = args.test
    if testset:
        testset = [int(x) for x in testset.split(",")]
    else:
        testset = list(range(40, 48))

    train_dataset = ImageSequenceHDF5Dataset(
        hdf5_root_path=args.dataset,
        indice=trainset,
        seq_len=args.seq_len,
        semi=args.semi,
        size=args.size,
        labelset=labelset,
        image_key=image_key,
    )
    test_dataset = ImageSequenceHDF5Dataset(
        hdf5_root_path=args.dataset,
        indice=testset,
        seq_len=args.seq_len,
        size=args.size,
        image_key=image_key,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KeyIntervalSegmenter(
        noise_std=args.noise_std, is_semi=args.semi
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999)
    )

    ntrain = len(train_dataset)
    seq_len = train_dataset.seq_len
    Z = torch.zeros(ntrain, seq_len, 2).float().to(device)
    z = torch.zeros(ntrain, seq_len, 2).float().to(device)
    outputs_cache = torch.zeros(ntrain, seq_len, 2).float().to(device)

    best_acc = -1.0
    best_epoch = 0
    best_model = None
    beta = args.beta

    for epoch in range(args.num_epochs):
        model.train()
        w = weight_schedule(
            epoch, args.max_epochs_ramp, args.ramp_mult, args.max_w
        )
        w = torch.tensor([w], device=device)
        last_file_idx = None
        hidden = None

        for i, (images, labels, file_idx, start_idx) in enumerate(
            train_loader
        ):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if file_idx != last_file_idx:
                hidden = model.init_hidden(images.size(0))
                last_file_idx = file_idx
            out, hidden = model(images, hidden)
            hidden = hidden.detach()
            zcomp = z[i * args.batch_size : (i + 1) * args.batch_size]
            loss, _, _ = temporal_loss_rnn(
                out, zcomp, w, labels,
                gamma=args.gamma, alpha=args.alpha, semi=args.semi
            )
            outputs_cache[i * args.batch_size : (i + 1) * args.batch_size] = out.detach()
            loss.backward()
            optimizer.step()

        Z = beta * Z + (1.0 - beta) * outputs_cache
        z = Z / (1.0 - beta ** (epoch + 1))

        val_loss, val_acc, acc_1, acc_0 = calc_metrics(
            model, test_loader,
            gamma=args.gamma, alpha=args.alpha, device=device
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model)
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1} val_acc={val_acc:.4f} "
                f"(best={best_acc:.4f} @ {best_epoch + 1})"
            )

    print(f"Best epoch: {best_epoch + 1}, best val_acc: {best_acc:.4f}")

    if best_model is not None:
        model = best_model

    predict_start, predict_end = 0, 200
    if args.predict_range:
        parts = args.predict_range.split("-")
        if len(parts) == 2:
            predict_start, predict_end = int(parts[0]), int(parts[1])

    for i in range(predict_start, predict_end):
        path = os.path.join(args.dataset, f"episode_{i}.hdf5")
        if os.path.exists(path):
            predict_and_write_hdf5(
                path, model, seq_len=args.seq_len, size=args.size,
                image_key=image_key, device=device
            )
            if (i - predict_start) % 20 == 0:
                print(f"Predicted labels for episode_{i}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output, "model.pt"))
        print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
