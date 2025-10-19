from functools import cached_property
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import fsolve

import consts


# plt.style.use(consts.PlotConfig.plt_style)  # Apply plot style


# function that desribe the time the reticulocytes spend in the bone marrow in seconds
def r_in_bm(c):
    return (0.065 * (3 / consts.conversion_factor) * c + 0.675) * 60 * 60 * 24


class HematopoiesisModel:
    def __init__(
        self,
        gamma_e=None,
        gamma_c=None,
        h_max=None,
        e_max=None,
        c_normalization_epo=None,
        a_max=None,
        k_a=None,
        d_max=None,
        k_d=None,
        adapt_max=0,
        adapt_k=consts.hbg_to_rbc(
            0.02
        ),  # acoording to the gap betwwen current hbg ang the running average
        adapt_memory=1 / 80000,  # 1 / seconds, about once a day
    ):
        if not gamma_e:
            gamma_e = consts.DEFAULT_MODEL_PARAMS["gamma_e"]
        if not gamma_c:
            gamma_c = consts.DEFAULT_MODEL_PARAMS["gamma_c"]
        if not h_max:
            h_max = consts.DEFAULT_MODEL_PARAMS["h_max"]
        if not e_max:
            e_max = consts.DEFAULT_MODEL_PARAMS["e_max"]
        if not c_normalization_epo:
            c_normalization_epo = consts.DEFAULT_MODEL_PARAMS["c_normalization_epo"]
        if not a_max:
            a_max = consts.DEFAULT_MODEL_PARAMS["a_max"]
        if not k_a:
            k_a = consts.DEFAULT_MODEL_PARAMS["k_a"]
        if not d_max:
            d_max = consts.DEFAULT_MODEL_PARAMS["d_max"]
        if not k_d:
            k_d = consts.DEFAULT_MODEL_PARAMS["k_d"]

        # Initialize all constants
        self.gamma_e = gamma_e
        self.gamma_c = gamma_c
        self.h_max = h_max
        self.e_max = e_max
        self.c_normalization_epo = c_normalization_epo
        self.a_max = a_max
        self.k_a = k_a
        self.d_max = d_max
        self.k_d = k_d
        self.adapt_max = adapt_max
        self.adapt_k = adapt_k
        self.adapt_memory = adapt_memory

    def a(self, e, a_max=None, k_a=None):
        # return a(E) [1/sec]
        if a_max is None:
            a_max = self.a_max
        if k_a is None:
            k_a = self.k_a
        return a_max * e / (k_a + e)

    def d(self, e, d_max=None, k_d=None):
        # return d(E) [1/sec]
        if d_max is None:
            d_max = self.d_max
        if k_d is None:
            k_d = self.k_d
        return d_max * e / (k_d + e)

    def gamma_r(self, c):
        # return gamma_R [1/sec]
        return 1 / r_in_bm(c)

    def adapt(self, c, c_avg):  # adaptaion to prolong anemia
        c_gap = c_avg - c
        return 1 + ((self.adapt_max * c_gap) / (self.adapt_k + np.abs(c_gap)))

    def dedt(self, c, r, e, h):
        # return dE/dt [mUnits/(ml*sec)]
        # eff_c = (1 - self.hypoxia_factor) * c * self.adapt(c, c_avg)
        eff_c = (1 - self.hypoxia_factor) * c
        return self.e_max * np.exp(-eff_c / self.c_normalization_epo) - e * (
            self.gamma_e * h
        )

    def dhdt(self, c, r, e, h):
        # return dH/dt [cells/sec]
        return self.a(e) * (1 - (h / self.h_max)) * h - self.d(e) * h

    def drdt(self, c, r, e, h):
        # return dR/dt [cells/sec]
        # return self.d(e) * h - GAMMA_R * r
        return self.d(e) * h - self.gamma_r(c) * r

    def dcdt(self, c, r, e, h):
        # return dC/dt [cells/sec]
        return self.gamma_r(c) * r - self.gamma_c * c

    def run_simulation(
        self,
        t_start,
        t_end,
        dt,
        c_initial,
        r_initial,
        e_initial,
        h_initial,
        hypoxia_factor,
        purturbation,
        show_progress=True,  # parameter to control tqdm progress bar printing
    ):
        # Set time
        num_steps = int((t_end - t_start) / dt)
        t = np.linspace(0, (num_steps - 1) * dt, num_steps)

        r_delay_steps = int(consts.R_DELAY / dt)  # Time steps of C production delay

        self.hypoxia_factor = hypoxia_factor

        # print the purturbation deatils, if the purturbation is not an empty
        # dictionary
        if purturbation["type"] != "none":
            print("Purturbation details:")
            print(f"Type: {purturbation['type']}")
            print(f"Time: {purturbation['time_days']} days")

            # Convert days to seconds and compute time steps
            t_purturbation = purturbation["time_days"] * 86400
            step_purturbation = np.round(t_purturbation / dt).astype(
                int
            )  # Convert to integer steps
        else:
            step_purturbation = []

        if purturbation["type"] == "blood_loss":
            volume_purturbation = purturbation["volume_liters"]
            c_purturbation = volume_purturbation * 5e12  # cells

        # Initialize arrays
        c = np.zeros(num_steps)  # level of RBCs. [cells]
        r = np.zeros(num_steps)  # number of reticulocytes. [cells]
        e = np.zeros(num_steps)  # level of EPO. [mU/mL]
        h = np.zeros(num_steps)  # level of CFU-E cells in BM. [cells]

        # Set initial conditions
        c[0] = c_initial
        r[: r_delay_steps + 1] = r_initial
        e[0] = e_initial
        h[0] = h_initial

        # The time loop
        for i in tqdm(range(1, num_steps), disable=not show_progress):
            j = i - r_delay_steps
            if j > 0:
                r[i] = r[i - 1] + self.drdt(c[j - 1], r[j - 1], e[j - 1], h[j - 1]) * dt
            c[i] = c[i - 1] + self.dcdt(c[i - 1], r[i - 1], e[i - 1], h[i - 1]) * dt
            e[i] = e[i - 1] + self.dedt(c[i - 1], r[i - 1], e[i - 1], h[i - 1]) * dt
            h[i] = h[i - 1] + self.dhdt(c[i - 1], r[i - 1], e[i - 1], h[i - 1]) * dt
            if i in step_purturbation:
                if purturbation["type"] == "epo_injection":
                    print(e[i])
                    print(purturbation["epo_dose"])
                    e[i] += purturbation["epo_dose"]  # EPO injection
                    print(e[i])
                elif purturbation["type"] == "blood_loss":
                    c[i] -= c_purturbation  # blood loss
            # e[i] = min(e[i], consts.EPO_UPPER_LIMIT)  # EPO upper limit
            # e[i] = max(e[i], 0)  # EPO lower limit

        # Convert t to days
        t_days = t / 86400

        # -----------Helper ploting -----------------
        """
        # plot the adapt values
        _, ax = plt.subplots()
        ax.plot(t_days, self.adapt(c, c_avg))
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Adaptation")'
       

        # plot g_e
        _, ax = plt.subplots()
        ax.plot(t_days, g_e, color=consts.PlotConfig.e_color)
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("H(t)")
        """
        return SimulationResults(self, t_days=t_days, c=c, r=r, e=e, h=h)


class NullclinePlotter:
    def __init__(
        self,
        model: HematopoiesisModel,
        ss_values,
        R_vals,
        plot_normal_clines=True,
        normal_params=None,
    ):
        self.model = model
        self.ss_values = ss_values
        self.R_vals = R_vals
        self.plot_normal_clines = plot_normal_clines
        self.normal_params = normal_params

        # Extract the *current* steady-state (C, R, H)
        self.c_ss, self.r_ss, self.h_ss = ss_values["c"], ss_values["r"], ss_values["h"]

        # "Normal" steady-state
        self.c_ss_normal, self.r_ss_normal, self.h_ss_normal = (
            consts.C_0,
            consts.R_0,
            consts.H_0,
        )

    # Define helper functions for nullclines and dynamics
    def E_ss(self, *, C, H, params={}):
        c_normalization_epo = params.get(
            "c_normalization_epo", self.model.c_normalization_epo
        )
        e_max = params.get("e_max", self.model.e_max)
        gamma_e = params.get("gamma_e", self.model.gamma_e)
        return (e_max / (gamma_e * H)) * np.exp(-C / c_normalization_epo)

    def g(self, *, C, H, params={}):
        a_max = params.get("a_max", self.model.a_max)
        k_a = params.get("k_a", self.model.k_a)
        return self.model.a(self.E_ss(C=C, H=H, params=params), a_max=a_max, k_a=k_a)

    def j(self, *, C, H, params={}):
        d_max = params.get("d_max", self.model.d_max)
        k_d = params.get("k_d", self.model.k_d)
        return self.model.d(self.E_ss(C=C, H=H, params=params), d_max=d_max, k_d=k_d)

    def H_nullcline(self, C, params={}):
        h_max = params.get("h_max", self.model.h_max)

        H_guess = self.h_ss_normal

        def eq_for_H(H):
            lhs = self.g(H=H, C=C, params=params) * (1.0 - H / h_max)
            rhs = self.j(H=H, C=C, params=params)
            return lhs - rhs

        # We do 1D root-finding in H for each fixed C
        sol = fsolve(eq_for_H, H_guess, xtol=1e-12)
        if sol[0] < 0:
            print("H nullcline negative value")
            return np.nan
        # print("Found a solution for H nullcline")
        return sol[0]

    def C_nullcline(self, R, params=None):
        gamma_c = params.get("gamma_c", self.model.gamma_c)

        def eq_for_C(Cguess):
            return self.model.gamma_r(Cguess) * R - gamma_c * Cguess

        # 1D root-finding in C for each fixed H
        C_guess = self.c_ss_normal
        sol = fsolve(eq_for_C, C_guess)
        if sol[0] < 0:
            return np.nan
        return sol[0]

    def R_nullcline(self, *, C, R, params={}):
        h_max = params.get("h_max", self.model.h_max)

        def eq_for_R(H):
            lhs = H * self.j(C=C, H=H, params=params)
            rhs = self.model.gamma_r(C) * R
            return lhs - rhs

        # We do 1D root-finding in H for each fixed C
        H_guess = self.h_ss_normal
        sol = fsolve(eq_for_R, H_guess)
        if sol[0] < 0 or sol[0] > h_max:
            return np.nan
        return sol[0]

    def dH_dt(self, H, C, R, params={}):
        h_max = params.get("h_max", self.model.h_max)
        return (
            self.g(C=C, H=H, params=params) * H * (1 - H / h_max)
            - self.j(C=C, H=H, params=params) * H
        )

    def dC_dt(self, H, C, R):
        return self.model.gamma_r(C) * R - self.model.gamma_c * C

    # Helper function to plot nullclines
    def plot_nullclines_for_R(
        self,
        ax,
        R,
        params={},
        normal_state=False,
        label_suffix="",
    ):
        linestyle = "--" if normal_state else "-"

        # dH/dt=0
        ax.plot(
            self.C_vals,
            [self.H_nullcline(C, params=params) for C in self.C_vals],
            label=f"dH/dt = 0{label_suffix}",
            linestyle=linestyle,
            color=consts.PlotConfig.h_color,
        )
        # dR/dt=0
        ax.plot(
            self.C_vals,
            [self.R_nullcline(C=C, R=R, params=params) for C in self.C_vals],
            label=f"dR/dt = 0{label_suffix}",
            linestyle=linestyle,
            color=consts.PlotConfig.r_color,
        )
        # dC/dt=0
        ax.plot(
            # Broadcast C_nullcline over the length of H_vals
            [self.C_nullcline(R, params=params)]
            * len(self.H_vals),  # same R => single root
            self.H_vals,
            label=f"dC/dt = 0{label_suffix}",
            linestyle=linestyle,
            color=consts.PlotConfig.c_color,
        )

    @cached_property
    def C_vals(self):
        # Decide on the range for C and H
        if self.plot_normal_clines:
            # Expand domain to include both current and normal SS points
            C_min = min(self.c_ss, self.c_ss_normal) - 0.5e13
            C_max = max(self.c_ss, self.c_ss_normal) + 0.5e13
            C_vals = np.linspace(C_min, C_max, 200)
        else:
            # Only center around the current SS
            C_vals = np.linspace(self.c_ss - 0.5e13, self.c_ss + 0.5e13, 200)

        return C_vals

    @cached_property
    def H_vals(self):
        # Decide on the range for C and H
        if self.plot_normal_clines:
            # Expand domain to include both current and normal SS points

            # We also need H_min/H_max that account for both the normal & current R_nullclines
            # for the first R in R_vals (or for normal R).
            # We'll compute R_nullcline over that range for each scenario, then pick min & max.
            R_first = self.R_vals[0]
            rvals_current = [self.R_nullcline(C=C, R=R_first) for C in self.C_vals]
            rvals_normal = []
            if self.normal_params:
                rvals_normal = [
                    self.R_nullcline(C=C, R=self.r_ss_normal, params=self.normal_params)
                    for C in self.C_vals
                ]
            # Combine with the actual H steady states, so they're inside the
            # range
            h_candidates = rvals_current + rvals_normal + [self.h_ss, self.h_ss_normal]
            # Remove NaNs
            h_candidates = [h for h in h_candidates if not np.isnan(h)]

            H_min = min(h_candidates) - 0.5e12
            H_max = max(h_candidates) + 0.5e12
            H_vals = np.linspace(H_min, H_max, 200)
        else:
            # Only center around the current SS
            H_vals = np.linspace(self.h_ss - 0.5e12, self.h_ss + 0.5e12, 200)

        return H_vals

    @cached_property
    def C_vals_sparse(self):
        # Prepare a grid for the vector field (quiver)
        return np.linspace(self.C_vals[0], self.C_vals[-1], 20)

    @cached_property
    def H_vals_sparse(self):
        return np.linspace(self.H_vals[0], self.H_vals[-1], 20)

    @cached_property
    def sparse_grid(self):
        return np.meshgrid(self.C_vals_sparse, self.H_vals_sparse)

    @cached_property
    def magnitude_sparse(self):
        return np.sqrt(self.dH_sparse_unnormalized**2 + self.dC_sparse_unnormalized**2)

    @cached_property
    def C_grid_sparse(self):
        return self.sparse_grid[0]

    @cached_property
    def H_grid_sparse(self):
        return self.sparse_grid[1]

    @cached_property
    def dH_sparse_unnormalized(self):
        # Evaluate the direction field
        return self.dH_dt(self.H_grid_sparse, self.C_grid_sparse, self.r_ss)

    @cached_property
    def dH_sparse(self):
        # Normalize to get direction arrows
        return self.dH_sparse_unnormalized / self.magnitude_sparse

    @cached_property
    def dC_sparse_unnormalized(self):
        # Evaluate the direction field
        return self.dC_dt(self.H_grid_sparse, self.C_grid_sparse, self.r_ss)

    @cached_property
    def dC_sparse(self):
        # Normalize to get direction arrows
        return self.dC_sparse_unnormalized / self.magnitude_sparse

    def plot_nullclines(self):
        # If multiple R values => multiple subplots
        if len(self.R_vals) > 1:
            fig, axs = plt.subplots(1, len(self.R_vals), figsize=(15, 5), sharey=True)
            for i, R in enumerate(self.R_vals):
                ax = axs[i]
                self.plot_nullclines_for_R(ax, R)
                ax.quiver(
                    self.C_grid_sparse,
                    self.H_grid_sparse,
                    self.dC_sparse,
                    self.dH_sparse,
                    angles="xy",
                    scale=30,
                    color="gray",
                    alpha=0.6,
                )
                # Current SS
                ax.plot(
                    self.c_ss,
                    self.h_ss,
                    "o",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markeredgewidth=2,
                    label="Steady state",
                )
                ax.set_ylim([self.H_vals[0], self.H_vals[-1]])
                ax.set_xlabel("C (RBC Count)")
                ax.set_ylabel("H (HSC Count)")
                ax.set_title(f"Phase Portrait (R={R:.2e})")
                ax.legend()
                ax.grid()
            fig.suptitle(
                "Phase Portrait with Nullclines for Different R Values", fontsize=16
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])

        else:
            # Single subplot
            R = self.R_vals[0]
            plt.figure(figsize=(12, 8))

            # Plot disease/current state nullclines
            self.plot_nullclines_for_R(plt.gca(), R)

            # Vector field
            plt.quiver(
                self.C_grid_sparse,
                self.H_grid_sparse,
                self.dC_sparse,
                self.dH_sparse,
                angles="xy",
                scale=30,
                color="gray",
                alpha=0.6,
            )
            # Mark the current steady state
            plt.plot(
                self.c_ss,
                self.h_ss,
                "o",
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=2,
                label="Steady state",
            )

            # If requested, overlay the "normal" nullclines & normal SS
            if self.plot_normal_clines and self.normal_params is not None:
                self.plot_nullclines_for_R(
                    plt.gca(),
                    self.r_ss_normal,
                    params=self.normal_params,
                    normal_state=True,
                    label_suffix=" (normal params)",
                )
                # Mark the normal intersection
                plt.plot(
                    self.c_ss_normal,
                    self.h_ss_normal,
                    "o",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markeredgewidth=2,
                    label="Normal steady state",
                )

            plt.xlabel("C (RBC Count)")
            plt.ylabel("H (HSC Count)")
            plt.title("Phase Portrait with Nullclines for dH/dt, dC/dt, and dR/dt")
            plt.legend()
            plt.grid()

        plt.show()


class SimulationResults:
    def __init__(self, model: HematopoiesisModel, t_days, c, r, e, h):
        self.model = model
        self.t_days = t_days
        self.c = c
        self.r = r
        self.e = e
        self.h = h

    @property
    def hbg(self):
        return consts.rbc_to_hbg(self.c)

    # @property
    # def hbg_avg(self):
    #    return consts.rbc_to_hbg(self.c_avg)

    def calculate_steady_state(self, print_values=True):
        # Calculate the steady state
        c_steady = self.c[-1]
        r_steady = self.r[-1]
        e_steady = self.e[-1]
        h_steady = self.h[-1]
        hbg_steady = consts.rbc_to_hbg(c_steady)

        if print_values:
            # print the steady state values
            print("\nSteady state values:")
            print(f"Hemoglobin steady state: {hbg_steady:.2f}")
            print(
                f"EPO steady state: {e_steady:.2f}, which is {(e_steady * consts.GAMMA_E * h_steady) / consts.E_0 * 100:.2f}% of maximal EPO"
            )
            print(
                f"HSCc steady state: {h_steady:.2e}, which is {h_steady / consts.H_MAX * 100:.2f}% of baseline"
            )
            print(f"RBC steady state: {c_steady:.2e}")
            print(f"Reticulocytes in bone marrow steady state: {r_steady:.2e}")
            # print the reticulocytes in percentage out of all RBCs
            gamma_R_ss = 1 / r_in_bm(c_steady)
            gamma_Rb = 1 / (
                3.9e5 - r_in_bm(c_steady)
            )  # Reticulocytes transition rate to RBCs in the blood [1/sec]
            reticulocytes_percentage = (
                (r_steady / c_steady) * (gamma_R_ss / gamma_Rb) * 100
            )
            print(f"\t Reticulocytes percentage: {reticulocytes_percentage:.2f}%")

        return c_steady, r_steady, e_steady, h_steady, hbg_steady

    def plot(self, x_higher_limit=None, ax=None):
        # Plot the results

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.figure

        # ax = plt.subplot(111)
        ax2 = ax.twinx()

        # Add grid lines
        ax.grid()

        ax.plot(self.t_days, self.hbg, color=consts.PlotConfig.c_color)
        ax2.plot(self.t_days, self.e, color=consts.PlotConfig.e_color)

        # Set the y axis limits
        ax.set_ylim(min(self.hbg) - 0.5, max(self.hbg) + 1)  # Hemoglobin y-axis range
        ax2.set_ylim(min(self.e) - 10, max(self.e) + 10)  # EPO y-axis range

        if x_higher_limit:
            ax.set_xlim(0, x_higher_limit)

        # Add legend
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Hemoglobin [g/dL]", color=consts.PlotConfig.c_color)
        ax2.set_ylabel("EPO [U/L]", color=consts.PlotConfig.e_color)

        ax.set_title("Hemoglobin and EPO levels over time")

        return fig, (ax, ax2)

    def plot_all(self, dt, t_end, purturbation):
        # create plot with 4 curve for each variable: h, c, r, e.

        print(purturbation["type"])

        if purturbation["type"] != "none":
            t_purturbation = purturbation["time_days"] * 86400
            t_start_plot = int((t_purturbation - (10 * 86400)) / dt)
            t_end_plot = int((t_purturbation + (100 * 86400)) / dt)
        else:
            t_start_plot = 0
            t_end_plot = int(t_end / dt)

        # Create the figure and the first y-axis
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # Create secondary y-axes for each variable
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax4 = ax1.twinx()

        # Offset the third and fourth y-axes to prevent overlap
        ax3.spines["right"].set_position(("outward", 60))
        ax4.spines["right"].set_position(("outward", 120))

        # Plot on the first y-axis (HSCc)
        ax1.plot(
            self.t_days[t_start_plot:t_end_plot],
            self.e[t_start_plot:t_end_plot],
            consts.PlotConfig.e_color,
            label="Erythropoietin",
        )
        ax1.set_ylabel("Erythropoietin", color=consts.PlotConfig.e_color)
        ax1.tick_params(axis="y", colors=consts.PlotConfig.e_color)

        # Plot on the second y-axis (Reticulocytes)
        ax2.plot(
            self.t_days[t_start_plot:t_end_plot],
            self.h[t_start_plot:t_end_plot],
            consts.PlotConfig.h_color,
            label="HSC",
        )
        ax2.set_ylabel("HSC", color=consts.PlotConfig.h_color)
        ax2.tick_params(axis="y", colors=consts.PlotConfig.h_color)

        # Plot on the fourth y-axis (Erythropoietin)
        ax3.plot(
            self.t_days[t_start_plot:t_end_plot],
            self.r[t_start_plot:t_end_plot],
            consts.PlotConfig.r_color,
            label="Reticulocytes",
        )
        ax3.set_ylabel("Reticulocytes", color=consts.PlotConfig.r_color)
        ax3.tick_params(axis="y", colors=consts.PlotConfig.r_color)

        # Plot on the third y-axis (RBC)
        ax4.plot(
            self.t_days[t_start_plot:t_end_plot],
            self.c[t_start_plot:t_end_plot],
            consts.PlotConfig.c_color,
            label="RBC",
        )
        ax4.set_ylabel("RBC", color=consts.PlotConfig.c_color)
        ax4.tick_params(axis="y", colors=consts.PlotConfig.c_color)

        if purturbation["type"] != "none":
            # Add dashed line at the time of the perturbation
            ax1.axvline(
                t_purturbation / 86400,
                color="black",
                linestyle="--",
                label="Perturbation",
            )

        # Add a y-axis limits for each variable
        ax1.set_ylim(0, 50)  # EPO
        ax2.set_ylim(2e12, 12e13)  # HSC
        ax3.set_ylim(5e11, 20e11)  # Reticulocytes
        ax4.set_ylim(5e12, 30e12)  # RBC

        # Add x-axis labels and title
        ax1.set_xlabel("Time [days]")
        plt.title("Temporal Dynamics of the variables")

        # Add a grid
        ax1.grid(True)

        plt.tight_layout()
        plt.show()
