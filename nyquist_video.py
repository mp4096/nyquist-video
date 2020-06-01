import argparse
import logging

import tomlkit
import numpy as np
import matplotlib.pyplot as plt


class Ctx:
    def __init__(self, config):
        self.config = config
        self.frequency_vec = None
        self.response_vec = None

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
            -self.frequency_vec * 1.0j * self.config["xfer_fun"]["time_delay"]
        )

    def plot_frame(self, idx, filename):
        fig = plt.figure(figsize=self.config["plotting"]["fig_size"])
        plt.plot(self.response_vec.real, self.response_vec.imag)
        print(self.response_vec[idx])
        plt.plot(
            [-1.0, self.response_vec[idx].real], [0.0, self.response_vec[idx].imag]
        )
        if self.config["plotting"]["axis_equal"]:
            plt.axis("equal")
        plt.xlim(*self.config["plotting"]["xlim"])
        plt.xlabel(self.config["plotting"]["xlabel"])
        plt.ylim(*self.config["plotting"]["ylim"])
        plt.ylabel(self.config["plotting"]["ylabel"])
        plt.grid(self.config["plotting"]["grid"])
        fig.savefig(
            filename,
            bbox_inches="tight",
            pad_inches=self.config["plotting"]["pad_inches"],
            dpi=self.config["plotting"]["dpi"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Nyquist criterion video.")
    parser.add_argument("config_file", type=str, help="path to the config file")
    args = parser.parse_args()

    with open(args.config_file, "r", encoding="utf8") as f:
        s = f.read()
    config = tomlkit.parse(s)

    ctx = Ctx(config)
    ctx.compute_points()
    ctx.plot_frame(500, "bla.png")
