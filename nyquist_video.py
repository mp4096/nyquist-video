import argparse

import tomlkit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from tqdm import tqdm


class Ctx:
    def __init__(self, config):
        self.config = config
        self.frequency_vec = None
        self.response_vec = None
        self.ray_angle = None
        self.angle_change = None

    def compute_points(self):
        self.frequency_vec = np.logspace(
            self.config["frequency_range"]["log10_range_lim"][0],
            self.config["frequency_range"]["log10_range_lim"][1],
            self.config["frequency_range"]["num_points"],
        )
        self.response_vec = np.polyval(
            self.config["xfer_fun"]["numerator"], self.frequency_vec * 1.0j
        )
        self.response_vec /= np.polyval(
            self.config["xfer_fun"]["denominator"], self.frequency_vec * 1.0j
        )
        self.response_vec *= np.exp(
            self.frequency_vec * (-1.0j) *
            self.config["xfer_fun"]["time_delay"]
        )

    def plot_frame(self, idx, filename):
        fig, ax = plt.subplots(figsize=self.config["plotting"]["fig_size"])
        plt.plot(
            self.response_vec.real,
            self.response_vec.imag,
            label=self.config["plotting"]["response"]["label"],
            c=self.config["plotting"]["response"]["color"],
        )
        plt.plot(
            [-1.0, self.response_vec[idx].real],
            [0.0, self.response_vec[idx].imag],
            label=self.config["plotting"]["ray"]["label"],
            c=self.config["plotting"]["ray"]["color"],
        )
        if self.config["plotting"]["axis_equal"]:
            plt.axis("equal")
        plt.xlim(*self.config["plotting"]["xlim"])
        plt.xlabel(self.config["plotting"]["xlabel"])
        plt.ylim(*self.config["plotting"]["ylim"])
        plt.ylabel(self.config["plotting"]["ylabel"])
        plt.grid(self.config["plotting"]["grid"])
        plt.legend()
        smaller_angle = np.min(self.ray_angle[[0, idx]])
        larger_angle = np.max(self.ray_angle[[0, idx]])
        wedge = Wedge(
            (-1.0, 0.0),
            self.config["plotting"]["wedge"]["radius"],
            smaller_angle,
            larger_angle,
        )
        pc = PatchCollection(
            [wedge],
            facecolors=[self.config["plotting"]["wedge"]["color"]],
            alpha=0.4,
        )
        ax.add_collection(pc)
        fig.savefig(
            filename,
            bbox_inches="tight",
            pad_inches=self.config["plotting"]["pad_inches"],
            dpi=self.config["plotting"]["dpi"],
        )
        plt.close()

    def compute_ray_angle(self):
        assert self.frequency_vec is not None
        assert self.response_vec is not None
        delta_angle = np.zeros(self.frequency_vec.shape)
        for i in range(1, len(self.frequency_vec)):
            p1 = self.response_vec[i - 1] + 1.0
            p2 = self.response_vec[i] + 1.0
            delta_angle[i] = np.arctan2(
                p1.real * p2.imag - p1.imag * p2.real,
                p1.real * p2.real + p1.imag * p2.imag,
            )
        self.angle_change = np.rad2deg(np.cumsum(delta_angle))
        self.ray_angle = self.angle_change + \
            np.angle(self.response_vec[0], deg=True)
        assert self.angle_change.shape == self.frequency_vec.shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Nyquist criterion video.")
    parser.add_argument("config_file", type=str,
                        help="path to the config file")
    args = parser.parse_args()

    with open(args.config_file, "r", encoding="utf8") as f:
        s = f.read()
    config = tomlkit.parse(s)

    ctx = Ctx(config)
    ctx.compute_points()
    ctx.compute_ray_angle()

    idxs = np.linspace(
        0,
        config["frequency_range"]["num_points"],
        config["animation"]["num_frames"],
        endpoint=False,
        dtype=int,
    )
    for i in tqdm(idxs):
        ctx.plot_frame(i, f"frame_{i:04d}.png")
