"""Rebuild the published AHCD training arrays and fixed split from the original archive.

Run this only when deliberately preparing a new shared data version. Normal model
notebooks read the included shared files without running this script.
"""
from pathlib import Path
from zipfile import ZipFile
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Resolve relative to this script so no personal computer path is needed.
data_folder = Path(__file__).resolve().parent
shared_folder = data_folder / "shared"
shared_folder.mkdir(parents=True, exist_ok=True)

# Refuse to silently replace the split already used by the team's models.
if any(shared_folder.glob("*.npz")):
    raise FileExistsError("Shared data already exist. Keep them for the current experiment. Use a separate folder for a new data version.")

# Read only the provided training portion. Keep the original test portion separate.
with ZipFile(data_folder / "AHCD.zip") as archive:
    with archive.open("csvTrainImages 13440x1024.csv") as file:
        pixels = pd.read_csv(file, header=None)

    with archive.open("csvTrainLabel 13440x1.csv") as file:
        labels = pd.read_csv(file, header=None).squeeze("columns")

assert pixels.shape == (13440, 1024)

assert len(labels) == 13440

assert not pixels.isna().any().any() and not labels.isna().any()

assert pixels.to_numpy().min() >= 0 and pixels.to_numpy().max() <= 255

assert np.equal(pixels.to_numpy(), np.floor(pixels.to_numpy())).all()

assert set(labels.unique()) == set(range(1, 29))

# Transpose each image once to restore its upright orientation, then add the channel.
images = np.ascontiguousarray(
    pixels.to_numpy(dtype="uint8").reshape(-1, 32, 32).transpose(0, 2, 1)[:, None, :, :]
)
labels = labels.to_numpy(dtype="int64") - 1
train_indices, validation_indices = train_test_split(
    np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42
)
np.savez_compressed(shared_folder / "training_data.npz", images=images, labels=labels)
np.savez_compressed(shared_folder / "split_indices.npz",
                    train_indices=train_indices, validation_indices=validation_indices)

(shared_folder / "label_mapping.json").write_text(
    json.dumps({str(i): i + 1 for i in range(28)}, indent=2) + "\n"
)
info = {
    "dataset": "AHCD",
    "source": "https://www.kaggle.com/datasets/mloey1/ahcd1",
    "authors_repository": "https://github.com/mloey/Arabic-Handwritten-Characters-Dataset",
    "source_archive": "AHCD.zip",
    "source_images": "csvTrainImages 13440x1024.csv",
    "source_labels": "csvTrainLabel 13440x1.csv",
    "images": 13440,
    "classes": 28,
    "shape_per_image": [1, 32, 32],
    "stored_dtype": "uint8",
    "pixel_range": [0, 255],
    "orientation": "reshape 32 by 32 then transpose each image once",
    "label_mapping": "internal labels 0 to 27 correspond to original labels 1 to 28",
    "validation_fraction": 0.2,
    "split_seed": 42,
    "split_method": "stratified image level split of the supplied training portion",
    "training_images": 10752,
    "validation_images": 2688,
    "test_policy": "Original test portion is separate and is not used by this notebook."
}
(shared_folder / "data_info.json").write_text(json.dumps(info, indent=2) + "\n")
print("Prepared shared images, labels, split indices and label mapping in", shared_folder)
