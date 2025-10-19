from functools import cached_property
import glob
from pathlib import Path
from typing import Optional
from matplotlib import pyplot as plt, rcParams
from tqdm.auto import tqdm
from IPython.display import display

import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance_nd

import consts
from Erythropoiesis_sim import HematopoiesisModel, NullclinePlotter, SimulationResults


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


class Disease:
    def __init__(
        self,
        name: str,
        modified_params: dict = {},
        hypoxia_factor: float = 0,
        perturbation: dict = {"type": "none"},
    ):
        self.name = name
        self.modified_params = modified_params
        # Define simulation parameters - hypoxia
        self.hypoxia_factor = hypoxia_factor
        # Define simulation parameters - perturbation
        self.perturbation = perturbation

        self.results: Optional[SimulationResults] = None
        self.steady_state_values: Optional[dict] = None
        self.simulation_data: Optional[pd.DataFrame] = None

        self.params = consts.DEFAULT_MODEL_PARAMS.copy()
        self.params.update(modified_params)

        # Define initial conditions
        self.r_initial = consts.R_0  # initial reticulocytes [cells]
        self.e_initial = consts.E_0  # initial EPO [Units/Liter]
        self.h_initial = consts.H_0  # initial HSCc [cells]
        self.c_initial = consts.C_0  # initial RBC [cells]

        # Define simulation parameters - time
        self.t_start = 0
        self.t_end = 5e7
        self.dt = 1000

    @cached_property
    def model(self):
        return HematopoiesisModel(**self.params)

    def run_simulation(self):
        # Run simulation
        self.results = self.model.run_simulation(
            t_start=self.t_start,
            t_end=self.t_end,
            dt=self.dt,
            c_initial=self.c_initial,
            r_initial=self.r_initial,
            e_initial=self.e_initial,
            h_initial=self.h_initial,
            hypoxia_factor=self.hypoxia_factor,
            purturbation=self.perturbation,
        )

        # Calculate the steady state
        (
            c_steady_state,
            r_steady_state,
            e_steady_state,
            h_steady_state,
            hbg_steady_state,
        ) = self.results.calculate_steady_state()

        self.steady_state_values = {
            "h": h_steady_state,
            "r": r_steady_state,
            "c": c_steady_state,
            "e": e_steady_state,
            "hbg": hbg_steady_state,
        }

    def random_model(self, sd_params: dict = {}):
        # Sample model parameters
        gamma_c = np.random.lognormal(
            np.log(self.params["gamma_c"]), sd_params.get("gamma_c", 0.2)
        )
        gamma_e = np.random.lognormal(
            np.log(self.params["gamma_e"]), sd_params.get("gamma_e", 0.2)
        )
        h_max = np.random.lognormal(
            np.log(self.params["h_max"]), sd_params.get("h_max", 0.3)
        )
        e_max = np.random.lognormal(
            np.log(self.params["e_max"]), sd_params.get("e_max", 0.2)
        )
        c_normalization = np.random.lognormal(
            np.log(self.params["c_normalization_epo"]),
            sd_params.get("c_normalization_epo", 0.05),
        )
        a_max = np.random.lognormal(
            np.log(self.params["a_max"]), sd_params.get("a_max", 0.1)
        )
        d_max = np.random.lognormal(
            np.log(self.params["d_max"]), sd_params.get("d_max", 0.1)
        )
        k_a = np.random.lognormal(np.log(self.params["k_a"]), sd_params.get("k_a", 0.2))
        k_d = np.random.lognormal(np.log(self.params["k_d"]), sd_params.get("k_d", 0.2))

        return HematopoiesisModel(
            gamma_c=gamma_c,
            gamma_e=gamma_e,
            h_max=h_max,
            e_max=e_max,
            c_normalization_epo=c_normalization,
            a_max=a_max,
            k_a=k_a,
            d_max=d_max,
            k_d=k_d,
        )

    def simulate_population_variability(
        self,
        n_samples: int,
        sd_params: dict = {},
        verbose=False,
    ):
        """
        Simulates population variability in hematopoiesis model using random sampling of parameters.

        Parameters can be overridden by passing custom values (e.g., H_MAX=1e12).

        Returns:
            pd.DataFrame: DataFrame containing steady-state values for all samples.
        """

        # Create a dataframe for results
        columns = ["h", "r", "c", "e", "hbg"]
        self.simulation_data = pd.DataFrame(
            columns=columns, index=range(n_samples)
        ).apply(pd.to_numeric)

        for i in tqdm(range(n_samples)):
            # Create and simulate the model
            results = self.random_model(sd_params).run_simulation(
                t_start=self.t_start,
                t_end=self.t_end,
                dt=self.dt,
                c_initial=self.c_initial,
                r_initial=self.r_initial,
                e_initial=self.e_initial,
                h_initial=self.h_initial,
                hypoxia_factor=self.hypoxia_factor,
                purturbation=self.perturbation,
                show_progress=False,
            )

            # Store steady state results
            self.simulation_data.loc[i, ["c", "r", "e", "h", "hbg"]] = (
                results.calculate_steady_state(print_values=verbose)
            )

    def print_stats(self):
        mean_hbg = self.simulation_data["hbg"].mean()
        std_hbg = self.simulation_data["hbg"].std()
        mean_EPO = self.simulation_data["e"].mean()
        std_EPO = self.simulation_data["e"].std()
        print(f"Mean Hb: {mean_hbg:.2f} ± {std_hbg:.2f}")
        print(f"Mean logEPO: {mean_EPO:.2f} ± {std_EPO:.2f}")

    def compare_to_exp_data(self, save_fig: bool = False):
        # Use predefined color and marker
        disease_color = consts.disease_colors.get(self.name, "black")
        _marker = consts.disease_markers.get(self.name, "o")

        path = (
            str(
                Path(
                    "~/Library/Mobile Documents/com~apple~CloudDocs/Herut/"
                    "Python/Homeostasis_feedback_loops/EPO_Hb"
                ).expanduser()
            )
            + "/"
        )
        files_names = glob.glob(path + self.name + "*.xlsx")

        bioassay_color = adjust_lightness(disease_color, amount=1.4)

        Hb_all = []
        log_EPO_all = []

        fig, ax = plt.subplots(figsize=(6, 4))
        s = 10  # Marker size

        # --- Global style settings ---
        rcParams["font.family"] = "Arial"
        rcParams["axes.titlesize"] = 12
        rcParams["axes.labelsize"] = 12
        rcParams["xtick.labelsize"] = 10
        rcParams["ytick.labelsize"] = 10
        rcParams["legend.fontsize"] = 8

        # Flags to avoid repeated labels in legend
        bioassay_plotted = False
        immunoassay_plotted = False

        for file_name in files_names:
            data = pd.read_excel(file_name)
            Hb = data["Hb"].values
            log_EPO = data["logEPO"].values

            Hb_all.extend(Hb)
            log_EPO_all.extend(log_EPO)

            data_type = file_name.split("_")[-2]

            if data_type == "bioassay":
                color = bioassay_color
                label = "Bioassay experiments" if not bioassay_plotted else None
                bioassay_plotted = True
            else:
                color = disease_color
                label = "Immunoassay experiments" if not immunoassay_plotted else None
                immunoassay_plotted = True

            ax.scatter(Hb, log_EPO, marker="o", color=color, label=label, s=s)

        if self.name == "norm":
            m, b = np.polyfit(Hb_all, log_EPO_all, 1)
            print(f"log(EPO) = {m:.3f} * Hb + {b:.3f}")

        # Simulation
        if "hbg" in self.simulation_data:
            hbg = self.simulation_data["hbg"]
            e = self.simulation_data["e"].values
        else:
            hbg = consts.rbc_to_hbg(self.simulation_data["c"])
            e = self.simulation_data["e"]

        ax.scatter(
            hbg,
            np.log10(e),
            marker="o",
            color=adjust_lightness(disease_color, 0.4),
            label="Simulated",
            s=s,
        )

        # Reference line
        h_line = np.linspace(min(Hb_all), max(Hb_all), 100)
        log_epo = -0.137 * h_line + 2.947
        ax.plot(h_line, log_epo, color="black", label="Normal ref")

        ax.set_xlabel("Hemoglobin (g/dL)")
        ax.set_ylabel("log(EPO) (mU/ml)")
        ax.set_title(f"Hb vs. EPO in {self.name}")
        ax.legend()
        plt.tight_layout()
        # plt.show()

        if save_fig:
            # Define your desired folder
            save_dir = Path(
                "/Users/herutuzan/Documents/Research/Uri Alon/Hematopoasis/Plos computional biology"
            ).expanduser()
            save_dir.mkdir(parents=True, exist_ok=True)  # create it if it doesn’t exist

            # save figure
            fig = ax.figure  # get the parent Figure

            # Combine folder and filename
            save_path = save_dir / f"Fig{self.name}.tiff"

            fig.savefig(
                save_path,
                format="tiff",
                dpi=600,  # can change to 300 if needed
                bbox_inches="tight",  # trims whitespace
                pil_kwargs={
                    "compression": "tiff_lzw"
                },  # lossless compression, smaller file
            )

    def print_exp_data(self):
        # Print basic statistics of each exp data of the disease

        # Build path to data directory
        base_dir = Path(
            "~/Library/Mobile Documents/com~apple~CloudDocs/"
            "Herut/Python/Homeostasis_feedback_loops/EPO_Hb"
        ).expanduser()
        files = list(base_dir.glob(f"{self.name}*.xlsx"))

        # Collect per-file statistics
        rows = []
        for file_path in files:
            data = pd.read_excel(file_path)
            Hb = data["Hb"].dropna()
            log_EPO = data["logEPO"].dropna()
            EPO_data = (10**log_EPO).rename("EPO")
            rows.append(
                {
                    "File": file_path.name,
                    "N points": len(Hb),
                    "Mean Hb": Hb.mean(),
                    "Std Hb": Hb.std(),
                    "Mean EPO": (EPO_data.mean()),
                    "Std EPO": (EPO_data.std()),
                }
            )

        df_stats = pd.DataFrame(rows).round(3)
        display(df_stats)

    def calculate_wasserstein_distance(self):
        # Calculate the Wasserstein distance between the simulated and experimental Hb and EPO distributions

        # Build path to data directory
        base_dir = Path(
            "~/Library/Mobile Documents/com~apple~CloudDocs/"
            "Herut/Python/Homeostasis_feedback_loops/EPO_Hb"
        ).expanduser()
        files = list(base_dir.glob(f"{self.name}*.xlsx"))

        # Remove files with bioassay data (bioassay in their name)
        files = [f for f in files if "bioassay" not in f.name]

        Hb_exp = []
        log_EPO_exp = []

        for file_name in files:
            data = pd.read_excel(file_name)
            Hb = data["Hb"].values
            log_EPO = data["logEPO"].values

            Hb_exp.extend(Hb)
            log_EPO_exp.extend(log_EPO)

        if "hbg" in self.simulation_data:
            Hb_sim = self.simulation_data["hbg"]
        else:
            Hb_sim = consts.rbc_to_hbg(self.simulation_data["c"])
        log_EPO_sim = np.log10(self.simulation_data["e"])

        # Convert to numpy arrays
        Hb_exp = np.array(Hb_exp)
        log_EPO_exp = np.array(log_EPO_exp)
        Hb_sim = np.array(Hb_sim)
        log_EPO_sim = np.array(log_EPO_sim)

        # If disease is "norm", keep only paris where the hbg is 12 or higher
        if self.name == "norm":
            mask = Hb_sim >= 12
            Hb_sim = Hb_sim[mask]
            log_EPO_sim = log_EPO_sim[mask]
            mask_exp = Hb_exp >= 12
            Hb_exp = Hb_exp[mask_exp]
            log_EPO_exp = log_EPO_exp[mask_exp]

        # Normalize Hb and log EPO by dividing by the experimental standard.
        # There is no need to substract the mean, since the Wasserstein distance is insensitive to translations.
        Hb_exp_std = np.std(Hb_exp)
        Hb_exp = (np.array(Hb_exp)) / Hb_exp_std
        Hb_sim = (np.array(Hb_sim)) / Hb_exp_std

        log_EPO_exp_std = np.std(log_EPO_exp)
        log_EPO_exp = (np.array(log_EPO_exp)) / log_EPO_exp_std
        log_EPO_sim = (np.array(log_EPO_sim)) / log_EPO_exp_std

        # create two 2-d array of the hbg and log epo values for the simulated and experimental data
        exp_array = np.array([Hb_exp, log_EPO_exp]).T
        sim_array = np.array([Hb_sim, log_EPO_sim]).T

        w_distance = wasserstein_distance_nd(exp_array, sim_array)

        # Now calculate the distance between the simulated data and the
        # expirimental data of the healthy (norm) group
        if self.name != "norm":
            files_norm = list(base_dir.glob("norm*.xlsx"))

            Hb_exp_norm = []
            log_EPO_exp_norm = []

            for file_name in files_norm:
                data = pd.read_excel(file_name)
                Hb = data["Hb"].values
                log_EPO = data["logEPO"].values

                Hb_exp_norm.extend(Hb)
                log_EPO_exp_norm.extend(log_EPO)

            exp_array_norm = np.array([Hb_exp_norm, log_EPO_exp_norm]).T
            w_distance_norm = wasserstein_distance_nd(exp_array_norm, sim_array)

            # Normaliziation
            Hb_exp_norm_std = np.std(Hb_exp_norm)
            Hb_exp_norm = (np.array(Hb_exp_norm)) / Hb_exp_norm_std
            Hb_sim = (np.array(Hb_sim)) / Hb_exp_norm_std
            log_EPO_exp_norm_std = np.std(log_EPO_exp_norm)
            log_EPO_exp_norm = (np.array(log_EPO_exp_norm)) / log_EPO_exp_norm_std
            log_EPO_sim = (np.array(log_EPO_sim)) / log_EPO_exp_norm_std

            print(
                f"Wasserstein distance between simulated and normal experimental data: {w_distance_norm:.3f}"
            )

        print(
            f"Wasserstein distance between simulated and experimental data: {w_distance:.3f}"
        )

    def plot_nullclines(self):
        R_vals = [self.steady_state_values["r"]]

        nullcline_plotter = NullclinePlotter(
            self.model,
            ss_values=self.steady_state_values,
            R_vals=R_vals,
            plot_normal_clines=True,  # self.name != "norm",
            normal_params=consts.DEFAULT_MODEL_PARAMS.copy(),
        )
        nullcline_plotter.plot_nullclines()
        return nullcline_plotter

    def fit_epo_curve(self):
        # Print a linear fit for the relationship between log EPO and Hb
        m, b = np.polyfit(
            self.simulation_data["hbg"], np.log(self.simulation_data["e"]), 1
        )
        print(f"log(EPO) =  {m:.3f}* Hb + {b:.3f}")
        return m, b

    def save_simulation_data(self, path):
        # save results to csv with file name population_simulation.csv
        file_name = f"{self.name}_population_simulation.csv"
        self.simulation_data.to_csv(path + file_name)
        print(f"Saved simulation data for {self.name} to {file_name}")

    def load_simulation_data(self, path):
        # load results from csv with file name population_simulation.csv
        file_name = f"{self.name}_population_simulation.csv"
        self.simulation_data = pd.read_csv(path + file_name, index_col=0)
        self.simulation_data = self.simulation_data.apply(pd.to_numeric)
        print(f"Loaded simulation data for {self.name} from {file_name}")
