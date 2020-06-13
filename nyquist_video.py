import argparse
import logging
from pathlib import Path

import tomlkit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.collections import PatchCollection
from tqdm import tqdm


class Ctx:
    def __init__(self, config):
        logging.info("Creating context")
        self.config = config
        self.output_dir = Path(self.config["plotting"]["export"]["output_dir"])
        self.frequency_vec = None
        self.response_vec = None
        self.ray_angle = None
        self.angle_change = None
        assert (
            not self.output_dir.exists()
        ), f"Output dir '{self.output_dir}' exists. Please remove it before proceeding"
        logging.info("Creating output dir '%s'", self.output_dir)
        self.output_dir.mkdir(parents=True)

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
            self.frequency_vec * (-1.0j) * self.config["xfer_fun"]["time_delay"]
        )

    def plot_frame(self, idx: int):
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
        patches = [wedge]
        patches.extend(
            Circle((-1.0, 0.0), self.config["plotting"]["wedge"]["radius"])
            for _ in range(int(self.angle_change[idx] / 360.0))
        )
        pc = PatchCollection(
            patches,
            facecolors=[self.config["plotting"]["wedge"]["color"]],
            alpha=self.config["plotting"]["wedge"]["alpha"],
        )
        title = (
            self.config["plotting"]["title_string"]
            .replace("{frequency}", f"{self.frequency_vec[idx]:.2e}")
            .replace("{angle_change}", f"{self.angle_change[idx]:.2f}")
        )
        plt.title(title)
        ax.add_collection(pc)
        filename = self.config["plotting"]["export"]["frame_prefix"] + f"{idx:04d}.png"
        fig.savefig(
            self.output_dir / filename,
            bbox_inches="tight",
            pad_inches=self.config["plotting"]["export"]["pad_inches"],
            dpi=self.config["plotting"]["export"]["dpi"],
        )
        plt.close()

    def compute_ray_angle(self):
        assert self.frequency_vec is not None
        assert self.response_vec is not None
        delta_angle = np.zeros(self.frequency_vec.shape)
        p1 = self.response_vec[0:-1] + 1.0
        p2 = self.response_vec[1:] + 1.0
        delta_angle[1:] = np.arctan2(
            p1.real * p2.imag - p1.imag * p2.real, p1.real * p2.real + p1.imag * p2.imag
        )
        self.angle_change = np.rad2deg(np.cumsum(delta_angle))
        self.ray_angle = self.angle_change.copy()
        self.ray_angle += np.angle(self.response_vec[0], deg=True)
        assert self.angle_change.shape == self.frequency_vec.shape

    def plot_all_frames(self):
        indices = np.linspace(
            0,
            self.config["frequency_range"]["num_points"] - 1,
            self.config["animation"]["num_frames"],
            dtype=int,
        )
        for i in tqdm(indices):
            self.plot_frame(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Nyquist criterion video.")
    parser.add_argument("config_file", type=str, help="path to the config file")
    args = parser.parse_args()

    with open(args.config_file, "r", encoding="utf8") as f:
        s = f.read()
    config = tomlkit.parse(s)

    ctx = Ctx(config)
    ctx.compute_points()
    ctx.compute_ray_angle()
    ctx.plot_all_frames()
