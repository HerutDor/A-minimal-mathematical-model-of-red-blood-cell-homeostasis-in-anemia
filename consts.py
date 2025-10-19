# Constants
BLOOD_VOLUME = 5  # Blood volume [liters]
R_DELAY = 2.5 * 24 * 3600  # seconds. Delay of RBCs to be produced after EPO is produced
EPO_UPPER_LIMIT = 20000  # EPO upper limit [mU/mL]

# Conversion factors form hemoglobin to rbc cells and back
hbg_weight_per_rbc = 0.3e-10  # grams/rbc
rbc_per_gram_hbg = 1 / hbg_weight_per_rbc  # rbc/gram
conversion_factor = rbc_per_gram_hbg * BLOOD_VOLUME * 10  # rbc/gram * liters * 10


def rbc_to_hbg(rbc):
    return rbc / conversion_factor


def hbg_to_rbc(hbg):
    return hbg * conversion_factor


# Defult constants
LAMBDA_E = 4e-5  # Total EPO degradation rate [1/sec]
GAMMA_R = 6.6e-6  # Reticulocyte degradation rate [1/sec]
GAMMA_C = 1e-7  # RBC degradation rate [1/sec]
GAMMA_H = 2.7e6  # HSCc that diffrentiation to other blood cells [1/sec]
H_MAX = 1e13  # Maximum number of CFU-E cells that the bone marrow can support [cells]
E_MAX = 0.0354  # Maximum EPO production rate [mU/(ml*sec)]
C_NORMALIZATION = (
    5.273e12  # Amount of C(t) in which EPO is generated in 1/e the rate [cells]
)
A_MAX = 5.28e-06  # Maximum CFU-E prolifration rate [1/sec]
K_A = 30  # Half-saturation constant for CFU-E proliferation [mU/ml]
D_MAX = 1.5e-6  # Maximum CFU-E differentiation rate [1/sec]
K_D = 23  # Half-saturation constant for CFU-E differentiation [mU/ml]

# Defult initial conditions (Steady state)
H_0 = 6.51e12  # Initial number of HSC cells [cells]
GAMMA_E = LAMBDA_E / H_0  # EPO degradation rate per HSC cell [1/(sec * cell)]
E_0 = 7.87  # Initial EPO concentration [mU/ml]
R_0 = 7.72e11  # Initial number of reticulocytes [cells]
HBG_0 = 14.94  # Initial hemoglobin concentration [mU/ml]
C_0 = hbg_to_rbc(HBG_0)  # Initial number of RBC cells [cells]

DEFAULT_MODEL_PARAMS = {
    "gamma_e": GAMMA_E,
    "gamma_c": GAMMA_C,
    "h_max": H_MAX,
    "e_max": E_MAX,
    "c_normalization_epo": C_NORMALIZATION,
    "a_max": A_MAX,
    "k_a": K_A,
    "d_max": D_MAX,
    "k_d": K_D,
}

# ---------- Define parameters for each disease ----------
# Parameters for CKD
e_max_ckd = E_MAX / 140
c_normalization_ckd = C_NORMALIZATION * 20

# Parameters for aplastic anemia
h_max_aa = H_MAX / 7

# Parameters for hemolytic anemia
rbc_life_span = 25  # days
gamma_c_ha = 1 / (rbc_life_span * 24 * 60 * 60)  # RBC degradation rate [1/sec]

# Parameters for iron deficiency anemia
d_max_ida = D_MAX / 3
a_max_ida = A_MAX / 3

# Parameters for anemia of chronic disease
d_max_acd = D_MAX / 4
# e_max_acd = E_MAX / 1.1

# Parameters for bone marrow suppression (mainly chemotherapy)
a_max_bms = A_MAX / 3.1


### ---------- Styling definitions ----------
class PlotConfig:
    # https://jrnold.github.io/ggthemes/reference/tableau_color_pal.html
    # plt_style = "seaborn-v0_8"
    h_color = "#4E79A7"  # blue
    r_color = "#B07AA1"  # purple
    c_color = "#E15759"  # red
    e_color = "#59A14F"  # green


# --- Define Colors for Each Variable ---
h_color = PlotConfig.h_color
r_color = PlotConfig.r_color
c_color = PlotConfig.c_color
e_color = PlotConfig.e_color

# --- Define Colors for Each Disease ---
# Additional colors: #EDC948, #BAB0AC
disease_colors = {
    "norm": "#4E79A7",  # blue
    "IDA": "#E15759",  # red
    "ACD": "#B07AA1",  # purple
    "CKD": "#59A14F",  # green
    "AA": "#F28E2B",  # orange
    "HA": "#9C755F",  # brown
    "BMS": "#636363",  # gray
    "disease": "#FF9DA7",  # pink
    "Cancer": "#76B7B2",  # cyan
}

# --- Define Markers for Each Disease ---
# https://matplotlib.org/stable/api/markers_api.html
disease_markers = {
    "IDA": "^",  # triangle-up
    "ACD": "v",  # triangle-down
    "AIHA": "<",  # triangle-left
    "AA": ">",  # triangle-right
    "HA": "1",  # tri-down
    "CKD": "P",  # plus (filled)
    "chemo": "p",  # pentagon
    "Cancer": "*",  # star
    "norm": "X",  # x
}
