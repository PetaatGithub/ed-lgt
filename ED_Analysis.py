import numpy as np
from simsio import *
import pickle


def save_dictionary(dict, filename):
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(dict, outp, pickle.HIGHEST_PROTOCOL)
    outp.close


def load_dictionary(filename):
    with open(filename, "rb") as outp:
        return pickle.load(outp)


# ACQUIRE SIMULATION RESULTS
config_filename = "DeltaN_single_mass"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["N", "g"])

obs_list = [
    "energy",
    "entropy",
    "gamma",
    "plaq",
    "n_single_EVEN",
    "n_single_ODD",
    "n_single_ODD",
    "n_pair_EVEN",
    "n_tot_EVEN",
    "n_tot_ODD",
]

res = {}
res["params"] = vals
for kk, obs in enumerate(obs_list):
    res[obs] = np.zeros((vals["N"].shape[0], vals["g"].shape[0]))

for ii, N in enumerate(vals["N"]):
    for kk, g in enumerate(vals["g"]):
        res["energy"][ii][kk] = get_sim(ugrid[ii][kk]).res["energy"][0]
        for kk, obs in enumerate(obs_list[1:]):
            res[obs][ii][kk] = get_sim(ugrid[ii][kk]).res[obs]

save_dictionary(res, "DeltaN_single_mass.pkl")

"""
Ent_entropy = np.zeros((vals["mass"].shape[0], vals["gSU2"].shape[0]))
for ii in range(vals["mass"].shape[0]):
    for jj in range(vals["gSU2"].shape[0]):
        psi = extract_dict(ugrid[ii][jj], key="res", glob="psi")

        # energy = get_sim(ugrid[ii][jj]).res["energy"]
        # sim= get_sim(ugrid[ii][jj])
        # sim.link("psi")
        # psi= sim.load("psi", cache= True tiene in memoria)
        # sim["psi"]
        Ent_entropy[ii][jj] = entanglement_entropy(psi, loc_dim=30, partition=2)

with open("Results/Simulation_Dicts/ED_2x2_MG_5eigs.pkl", "rb") as dict:
    ED_data = pickle.load(dict)
ED_data["entropy"] = Ent_entropy

energy = np.zeros((30, 30, 5))
for ii in range(30):
    for jj in range(30):
        energy[ii][jj] = extract_dict(ugrid[ii][jj], key="res", glob="energy")

ED_data["energy"] = energy
fidelity = np.zeros((30, 29))
for ii in range(vals["mass"].shape[0]):
    for jj in range(vals["gSU2"].shape[0] - 1):
        fidelity[ii][jj] = np.abs(
            np.real(
                np.dot(
                    np.conjugate(extract_dict(ugrid[ii][jj], key="res", glob="psi")),
                    extract_dict(ugrid[ii][jj + 1], key="res", glob="psi"),
                )
            )
        )

"""
