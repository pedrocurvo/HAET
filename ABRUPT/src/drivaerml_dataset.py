from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class DrivAerMLStats:
    raw_pos_min: tuple[float] = (-40.0,)
    raw_pos_max: tuple[float] = (80.0,)
    surface_pressure_mean: tuple[float] = (-2.29772e02,)
    surface_pressure_std: tuple[float] = (2.69345e02,)
    surface_wallshearstress_mean: tuple[float, float, float] = (-1.20054e00, 1.49358e-03, -7.20107e-02)
    surface_wallshearstress_std: tuple[float, float, float] = (2.07670e00, 1.35628e00, 1.11426e00)
    volume_totalpcoeff_mean: tuple[float] = (1.71387e-01,)
    volume_totalpcoeff_std: tuple[float] = (5.00826e-01,)
    volume_velocity_mean: tuple[float, float, float] = (1.67909e01, -3.82238e-02, 4.07968e-01)
    volume_velocity_std: tuple[float, float, float] = (1.64115e01, 8.63614e00, 6.64996e00)
    volume_vorticity_logscale_mean: tuple[float, float, float] = (-1.47814e-02, 7.87642e-01, 2.81023e-03)
    volume_vorticity_logscale_std: tuple[float, float, float] = (5.45681e00, 5.77081e00, 5.46175e00)


class DrivAerMLDefaultSplitIDs:
    # fmt: off
    train = {
        1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 21, 23, 25, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
        100, 101, 102, 103, 104, 105, 106, 107, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
        125, 126, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 139, 140, 141, 143, 144, 145, 146, 147, 148, 149,
        151, 152, 153, 154, 155, 156, 157, 159, 160, 161, 162, 163, 164, 166, 168, 169, 170, 171, 172, 174, 175, 176,
        178, 179, 181, 182, 183, 184, 185, 186, 189, 190, 192, 193, 194, 195, 196, 198, 200, 201, 202, 204, 206, 209,
        212, 213, 214, 216, 217, 219, 220, 223, 224, 225, 227, 229, 230, 231, 232, 233, 235, 236, 237, 238, 239, 240,
        242, 243, 244, 245, 246, 249, 250, 251, 254, 255, 256, 257, 259, 261, 262, 264, 265, 266, 267, 268, 269, 270,
        272, 273, 274, 276, 277, 278, 279, 281, 283, 285, 286, 287, 288, 289, 292, 293, 294, 296, 297, 299, 300, 301,
        302, 304, 305, 306, 307, 308, 309, 310, 312, 313, 314, 315, 317, 318, 319, 320, 323, 326, 327, 330, 331, 332,
        333, 334, 335, 336, 338, 339, 340, 342, 343, 344, 345, 346, 347, 348, 349, 351, 353, 355, 356, 357, 358, 359,
        360, 361, 362, 365, 367, 368, 369, 371, 373, 374, 375, 377, 378, 379, 381, 383, 384, 385, 386, 388, 389, 391,
        392, 393, 394, 395, 396, 397, 398, 399, 400, 402, 404, 406, 407, 408, 409, 411, 412, 413, 414, 415, 416, 417,
        418, 419, 420, 421, 422, 425, 426, 427, 430, 431, 432, 433, 434, 435, 437, 438, 439, 440, 442, 443, 444, 445,
        446, 448, 449, 450, 451, 452, 453, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469,
        470, 471, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 488, 489, 490, 491, 492, 493, 494,
        495, 496, 497, 498, 499, 500,
    }
    val = {
        4, 22, 56, 109, 150, 165, 177, 191, 228, 234, 241, 247, 252, 253, 260, 271, 275, 298, 303, 311, 321, 324, 328,
        341, 352, 366, 380, 390, 401, 423, 441, 447, 454, 487,
    }
    test = {
        11, 12, 19, 20, 24, 26, 29, 41, 55, 59, 108, 124, 127, 133, 142, 158, 173, 180, 187, 188, 197, 199, 203, 205,
        207, 208, 210, 215, 222, 226, 258, 263, 280, 284, 290, 322, 337, 350, 354, 363, 372, 382, 387, 405, 410, 424,
        428, 429, 436, 472,
    }
    tutorial = {11}
    # The following design IDs are not available in the dataset. They are held back by the authors for testing purposes.
    hidden_val = {167, 211, 218, 221, 248, 282, 291, 295, 316, 325, 329, 364, 370, 376, 403, 473}
    # fmt: on


class DrivAerMLDataset(Dataset):
    """Dataset implementation for DrivAerML that supports both local and AISTORE storage.

    Args:
        split: Which split to use.
        root: path to the processed dataset, e.g. .../drivaerml_processed/subsampled_10x.
    """

    def __init__(self, root: str, split: str):
        super().__init__()
        self.root = Path(root).expanduser()
        if split == "train":
            design_ids = DrivAerMLDefaultSplitIDs.train
        elif split == "val":
            design_ids = DrivAerMLDefaultSplitIDs.val
        elif split == "test":
            design_ids = DrivAerMLDefaultSplitIDs.test
        elif split == "tutorial":
            design_ids = DrivAerMLDefaultSplitIDs.tutorial
        else:
            raise NotImplementedError
        # convert sets to list
        self.design_ids = sorted(design_ids)

    def __len__(self):
        return len(self.design_ids)

    @staticmethod
    def get_normalization_stats():
        return DrivAerMLStats()

    def _load(self, idx: int, filename: str) -> torch.Tensor:
        design_uri = self.root / f"run_{self.design_ids[idx]}"
        assert design_uri.exists(), f"{design_uri.as_posix()} does not exist"
        return torch.load(design_uri / filename, weights_only=True)

    def getitem_surface_position_vtp(self, idx: int) -> torch.Tensor:
        """Retrieves surface positions from the CFD mesh (num_surface_points, 3)"""
        return self._load(idx=idx, filename="surface_position_vtp.pt")

    def getitem_surface_position_stl(self, idx: int) -> torch.Tensor:
        """Retrieves surface positions from the STL file (num_surface_points, 3)"""
        return self._load(idx=idx, filename="surface_position_stl_resampled100k.pt")

    def getitem_surface_pressure(self, idx: int) -> torch.Tensor:
        """Retrieves surface pressures (num_surface_points, 1)"""
        return self._load(idx=idx, filename="surface_pressure.pt").unsqueeze(1)

    def getitem_surface_wallshearstress(self, idx: int) -> torch.Tensor:
        """Retrieves surface wallshearstress (num_surface_points, 3)"""
        return self._load(idx=idx, filename="surface_wallshearstress.pt")

    def getitem_volume_position(self, idx: int) -> torch.Tensor:
        """Retrieves volume position (num_volume_points, 3)"""
        return self._load(idx=idx, filename="volume_cell_position.pt")

    def getitem_volume_totalpcoeff(self, idx: int) -> torch.Tensor:
        """Retrieves volume pressures (num_volume_points, 1)"""
        return self._load(idx=idx, filename="volume_cell_totalpcoeff.pt").unsqueeze(1)

    def getitem_volume_velocity(self, idx: int) -> torch.Tensor:
        """Retrieves volume velocity (num_volume_points, 3)"""
        return self._load(idx=idx, filename="volume_cell_velocity.pt")

    def getitem_volume_vorticity(self, idx: int) -> torch.Tensor:
        """Retrieves volume vorticity (num_volume_points, 3)"""
        return self._load(idx=idx, filename="volume_cell_vorticity.pt")
