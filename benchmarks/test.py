import datasets
import torch
from pprint import pprint

ds = datasets.load_dataset(
    "proxima-fusion/constellaration",
    split="train",
    num_proc=4,
)
ds = ds.select_columns([c for c in ds.column_names
                        if c.startswith("boundary.")
                        or c.startswith("metrics.")])
ds = ds.filter(
    lambda x: x == 3,
    input_columns=["boundary.n_field_periods"],
    num_proc=4,
)
ml_ds = ds.remove_columns([
    "boundary.n_field_periods", "boundary.is_stellarator_symmetric",  # all same value
    "boundary.r_sin", "boundary.z_cos",  # empty
    "boundary.json", "metrics.json", "metrics.id",  # not needed
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_ds = ml_ds.with_format("torch", device=device)  # other options: "jax", "tensorflow" etc.

for batch in torch.utils.data.DataLoader(torch_ds, batch_size=4, num_workers=4):
    pprint(batch)
    break
