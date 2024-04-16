# %%
from simsio import *
from math import prod
from matplotlib import pyplot as plt
from ed_lgt.tools import save_dictionary, load_dictionary


@plt.FuncFormatter
def fake_log(x, pos):
    "The two args are the value and tick position"
    return r"$10^{%d}$" % (x)


"""
To extract simulations use
    op1) energy[ii][jj] = extract_dict(ugrid[ii][jj], key="res", glob="energy")
    op2) energy[ii][jj] = get_sim(ugrid[ii][jj]).res["energy"])
To acquire the psi file
    sim= get_sim(ugrid[ii][jj])
    sim.link("psi")
    psi= sim.load("psi", cache=True)
"""
# %%
# List of local observables
local_obs = [f"n_{s}{d}" for d in "xy" for s in "mp"]
local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
local_obs += ["X_Cross", "S2"]

BC_list = ["PBCxy"]
lattize_size_list = ["2x2", "3x2"]
for ii, BC in enumerate(BC_list):
    res = {}
    # define the observables arrays
    res["energy"] = np.zeros((len(lattize_size_list), 25), dtype=float)
    for obs in local_obs:
        res[obs] = np.zeros((len(lattize_size_list), 25), dtype=float)
    for jj, size in enumerate(lattize_size_list):
        # look at the simulation
        config_filename = f"Z2FermiHubbard/{BC}/{size}"
        match = SimsQuery(group_glob=config_filename)
        ugrid, vals = uids_grid(match.uids, ["U"])
        lvals = get_sim(ugrid[0]).par["model"]["lvals"]
        for kk, U in enumerate(vals["U"]):
            for obs in local_obs:
                res[obs][jj, kk] = np.mean(get_sim(ugrid[kk]).res[obs])
                res["energy"][jj, kk] = get_sim(ugrid[kk]).res["energies"][0] / (
                    prod(lvals)
                )
    save_dictionary(res, f"{BC}.pkl")
# %%
for obs in ["X_Cross", "N_pair", "N_single", "energy", "S2"]:
    print(obs)
    fig = plt.figure()
    plt.ylabel(rf"{obs}")
    plt.xlabel(r"U")
    plt.xscale("log")
    plt.grid()
    for ii, label in enumerate(lattize_size_list):
        plt.plot(vals["U"], res[obs][ii, :], "-o", label=label)
    plt.legend(loc=(0.05, 0.11))
    # plt.savefig(f"{obs}.pdf")
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set(xscale="log", yscale="log")
config_filename = f"Z2FermiHubbard/prova1"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["U"])
lvals = get_sim(ugrid[0]).par["model"]["lvals"]
entropies = []
for kk, U in enumerate(vals["U"]):
    C = get_sim(ugrid[kk]).res["C"]
    entropies.append(get_sim(ugrid[kk]).res["entropy"])
    ax.plot(C[:, 0], C[:, 1], "-o", label=f"U={U}")
ax.set(xlabel="r=|i-j|", ylabel="<SzSz>")
plt.legend()

fig1, ax1 = plt.subplots()
for kk, U in enumerate(vals["U"]):
    ax1.plot(np.arange(1, 5, 1), entropies[kk][:4], "-o", label=f"U={U}")
ax1.set(xlabel="A", ylabel="EE")
plt.legend()


# %%
def r_values(energy):
    energy = np.sort(energy)
    delta_E = np.zeros(energy.shape[0] - 1)
    r_array = np.zeros(energy.shape[0] - 1)
    for ii in range(energy.shape[0] - 1):
        delta_E[ii] = energy[ii + 1] - energy[ii]
    for ii in range(delta_E.shape[0]):
        if ii == 0:
            r_array[ii] = 1
        else:
            r_array[ii] = min(delta_E[ii], delta_E[ii - 1]) / max(
                delta_E[ii], delta_E[ii - 1]
            )
    return r_array, delta_E


# %%
config_filename = f"SU2/N8"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
# N=10 8951 1790:7160
# N=8 1105 276:828
# N=6 139  35:104
N = 8
if N == 10:
    size = 8951
elif N == 8:
    size = 1105
res = {
    "entropy": np.zeros((vals["g"].shape[0], vals["m"].shape[0], size)),
    "energy": np.zeros((vals["g"].shape[0], vals["m"].shape[0], size)),
    "overlap_V": np.zeros((vals["g"].shape[0], vals["m"].shape[0], size)),
    "overlap_PV": np.zeros((vals["g"].shape[0], vals["m"].shape[0], size)),
    "overlap_M": np.zeros((vals["g"].shape[0], vals["m"].shape[0], size)),
    # "delta_E": np.zeros((vals["g"].shape[0], vals["m"].shape[0], size-1)),
    "r_values": np.zeros((vals["g"].shape[0], vals["m"].shape[0], size - 1)),
}

for kk, g in enumerate(vals["g"]):
    for ii, m in enumerate(vals["m"]):
        res["indices"] = get_sim(ugrid[kk][ii]).res["indices"]
        res["entropy"][kk, ii, :] = get_sim(ugrid[kk][ii]).res["entropy"]
        res["energy"][kk, ii, :] = get_sim(ugrid[kk][ii]).res["energy"]
        res["overlap_V"][kk, ii, :] = get_sim(ugrid[kk][ii]).res["overlap_V"]
        res["overlap_PV"][kk, ii, :] = get_sim(ugrid[kk][ii]).res["overlap_PV"]
        res["overlap_M"][kk, ii, :] = get_sim(ugrid[kk][ii]).res["overlap_M"]
        res["r_values"][kk, ii, :], _ = r_values(res["energy"][kk, ii, :])
        print("===================================================")
        # print(g, m, np.mean(res["r_values"][kk, ii, 276:828]))
save_dictionary(res, f"N10m1g5.pkl")
rval = np.array([0.36756723061786156, 0.36002678652893033, 0.4107746332363976])
# %%
config_filename = f"SU2/N10"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
res = {
    "entropy": np.zeros((vals["g"].shape[0], vals["m"].shape[0], 8951)),
    "energy": np.zeros((vals["g"].shape[0], vals["m"].shape[0], 8951)),
    "r_values": np.zeros((vals["g"].shape[0], vals["m"].shape[0], 8950)),
}


means = np.zeros((vals["g"].shape[0], vals["m"].shape[0]))

for kk, g in enumerate(vals["g"]):
    for ii, m in enumerate(vals["m"]):
        res["entropy"][kk, ii, :] = get_sim(ugrid[kk][ii]).res["entropy"]
        res["energy"][kk, ii, :] = get_sim(ugrid[kk][ii]).res["energy"]
        res["r_values"][kk, ii, :], _ = r_values(res["energy"][kk, ii, :])
        means[kk, ii] = np.mean(res["r_values"][kk, ii, 1790:7160])
        print("===================================================")
        print(g, m, means[kk, ii])
res["r_means"] = means
save_dictionary(res, f"scars4.pkl")
# %%
fig, ax = plt.subplots()
img = ax.imshow(
    np.transpose(means),
    origin="lower",
    cmap="magma",
    extent=[-1, 1, -1, 1],
    vmin=0.3,  # Set the minimum value for color normalization
    vmax=0.5,  # Set the maximum value for color normalization
)
ax.set_ylabel(r"$m$")
ax.set_xlabel(r"$g$")
ax.set(yticks=[-1, 0, 1], xticks=[-1, 0, 1])
ax.xaxis.set_major_formatter(fake_log)
ax.yaxis.set_major_formatter(fake_log)

cb = fig.colorbar(
    img,
    ax=ax,
    aspect=20,
    location="right",
    orientation="vertical",
    pad=0.01,
)

fig, ax = plt.subplots()
img = ax.imshow(
    np.transpose(res["entropy"][:, :, 1]),
    origin="lower",
    cmap="magma",
    extent=[-1, 1, -1, 1],
)
ax.set_ylabel(r"$m$")
ax.set_xlabel(r"$g$")
ax.set(yticks=[-1, 0, 1], xticks=[-1, 0, 1])
ax.xaxis.set_major_formatter(fake_log)
ax.yaxis.set_major_formatter(fake_log)

cb = fig.colorbar(
    img,
    ax=ax,
    aspect=20,
    location="right",
    orientation="vertical",
    pad=0.01,
)


# %%
import csv

with open("energy_overlap_PV.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Energy", "Overlap Pol Vacuum"])  # Writing header row
    for energy, overlap in zip(res["energy"][0, 0, :], res["overlap_PV"][0, 0, :]):
        writer.writerow([energy, overlap])  # Writing data rows

# %%
import csv

with open("energy_overlap_V.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Energy", "Overlap Vacuum"])  # Writing header row
    for energy, overlap in zip(res["energy"][0, 0, :], res["overlap_V"][0, 0, :]):
        writer.writerow([energy, overlap])  # Writing data rows

# %%
for kk, g in enumerate(vals["g"]):
    for ii, m in enumerate(vals["m"]):
        small_entropy_pos = res["entropy"][kk, ii, :] < 2
        small_entropy = res["entropy"][kk, ii, small_entropy_pos]
        small_entropy_energy = res["energy"][kk, ii, small_entropy_pos]
        small_entropy_overlap_V = res["overlap_V"][kk, ii, small_entropy_pos]
        small_entropy_overlap_PV = res["overlap_PV"][kk, ii, small_entropy_pos]

        fig, ax = plt.subplots()
        ax.scatter(
            res["energy"][kk, ii, :],
            res["entropy"][kk, ii, :],
            s=10,
            facecolors="white",
            edgecolors="black",
            label=f"m={m}, g={g}",
        )
        """        
        ax.scatter(
            res["energy"][kk, ii, res["indices"]],
            res["entropy"][kk, ii, res["indices"]],
            s=15,
            facecolors="red",
            edgecolors="red",
            label=f"m={m}, g={g}",
        )"""
        ax.set(xlabel="Energy", ylabel="Bipartite Ent. Entropy")
        ax.legend()
        ax.grid()
        """        
        fig, ax = plt.subplots()
        ax.scatter(
            np.arange(1104),
            res["delta_E"][kk, ii, :],
            s=10,
            facecolors="white",
            edgecolors="red",
            label=f"m={m}, g={g}",
        )
        ax.set(xlabel="n", ylabel="delta E n", yscale="log")
        ax.legend()
        ax.grid()"""
        # ==============================
        fig, ax = plt.subplots()
        ax.scatter(
            res["energy"][kk, ii, :],
            res["overlap_M"][kk, ii, :],
            s=10,
            facecolors="white",
            edgecolors="blue",
            label=f"m={m}, g={g}",
        )
        """ax.scatter(
            res["energy"][kk, ii, res["indices"]],
            res["overlap_PV"][kk, ii, res["indices"]],
            s=15,
            facecolors="red",
            edgecolors="red",
            label=f"m={m}, g={g}",
        )"""
        ax.set(
            xlabel="Energy",
            ylabel="Overlap Meson",
            yscale="log",
            # xlim=[-2.5, 7.5],
            ylim=[1e-12, 1],
        )
        ax.legend()
        ax.grid()
        fig, ax = plt.subplots()
        ax.scatter(
            res["energy"][kk, ii, :],
            res["overlap_V"][kk, ii, :],
            s=10,
            facecolors="white",
            edgecolors="brown",
            label=f"m={m}, g={g}",
        )
        ax.scatter(
            res["energy"][kk, ii, res["indices"]],
            res["overlap_V"][kk, ii, res["indices"]],
            s=15,
            facecolors="red",
            edgecolors="red",
            label=f"m={m}, g={g}",
        )
        ax.set(
            xlabel="Energy",
            ylabel="Overlap Vacuum",
            yscale="log",
            ylim=[1e-12, 1],
            # xlim=[-17, 14],
        )
        ax.legend()
        ax.grid()
save_dictionary(res, "scars3.pkl")

# %%
for ii in range(res["energy"].shape[2]):
    print(res["energy"][0, 0, ii], res["entropy"][0, 0, ii])
# %%
for kk, g in enumerate(vals["g"]):
    for ii, m in enumerate(vals["m"]):
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].scatter(
            small_entropy_energy,
            small_entropy,
            s=10,
            facecolors="white",
            edgecolors="black",
            label=f"N=10, m={m}, g={g}",
        )
        ax[0].set(ylabel="Bip. Ent. Entropy", ylim=[0, 2.2])
        ax[0].legend()
        # ==============================
        ax[1].scatter(
            small_entropy_energy,
            small_entropy_overlap_V,
            s=10,
            facecolors="white",
            edgecolors="red",
            label=f"m={m}, g={g}",
        )
        ax[1].set(
            ylabel="Ov. Vacuum",
            yscale="log",
            ylim=[1e-12, 2],
        )
        # ==============================
        ax[2].scatter(
            small_entropy_energy,
            small_entropy_overlap_PV,
            s=10,
            facecolors="white",
            edgecolors="blue",
            label=f"m={m}, g={g}",
        )

        ax[2].set(
            xlabel="Energy", ylabel="Ov. Pol Vacuum", yscale="log", ylim=[1e-12, 2]
        )
        for jj, axis in enumerate(ax.flat):
            axis.grid()
# %%
config_filename = f"SU2/dynamics"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
res = {
    "entropy": get_sim(ugrid[0][0]).res["entropy"],
    "fidelity": get_sim(ugrid[0][0]).res["fidelity"],
}

fig, ax = plt.subplots()
start = 0
stop = 3
delta_n = 0.01
n_steps = int((stop - start) / delta_n)
ax.plot(np.arange(n_steps) * delta_n, res["entropy"])
ax.set(xlabel="Time", ylabel="Entanglement entropy")
ax.grid()

fig, ax = plt.subplots()
start = 0
stop = 3
delta_n = 0.01
n_steps = int((stop - start) / delta_n)
ax.plot(np.arange(n_steps) * delta_n, res["fidelity"])
ax.grid()
ax.set(xlabel="Time", ylabel="Fidelity")
# %%
config_filename = f"QED/prova"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
lvals = get_sim(ugrid[0]).par["model"]["lvals"]
entropies = []
a = []
for kk, g in enumerate(vals["g"]):
    entropies.append(float(get_sim(ugrid[kk]).res["entropy"]))
    a.append(float(get_sim(ugrid[kk]).res["C_px,py_C_py,mx_C_my,px_C_mx,my"]))
# %%
config_filename = f"QED/convergence"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m", "spin"])

res = {
    "entropy": np.zeros(
        (vals["g"].shape[0], vals["m"].shape[0], vals["spin"].shape[0])
    ),
    "plaq": np.zeros((vals["g"].shape[0], vals["m"].shape[0], vals["spin"].shape[0])),
}

for ii, g in enumerate(vals["g"]):
    for jj, m in enumerate(vals["m"]):
        for kk, spin in enumerate(vals["spin"]):
            res["entropy"][ii, jj, kk] = get_sim(ugrid[ii][jj][kk]).res["entropy"]
            res["plaq"][ii, jj, kk] = get_sim(ugrid[ii][jj][kk]).res[
                "C_px,py_C_py,mx_C_my,px_C_mx,my"
            ]
# %%
abs_convergence = 1e-3
rel_convergence = 1e-1
convergence = np.zeros((vals["g"].shape[0], vals["m"].shape[0]))
entropy = np.zeros((vals["g"].shape[0], vals["m"].shape[0]))
plaq = np.zeros((vals["g"].shape[0], vals["m"].shape[0]))
for ii, g in enumerate(vals["g"]):
    for jj, m in enumerate(vals["m"]):
        for kk, spin in enumerate(vals["spin"]):
            if kk > 0:
                abs_delta = np.abs(
                    res["plaq"][ii, jj, kk] - res["plaq"][ii, jj, kk - 1]
                )
                rel_delta = abs_delta / np.abs(res["plaq"][ii, jj, kk])
                if abs_delta < abs_convergence and rel_delta < rel_convergence:
                    print(g, m, spin)
                    convergence[ii, jj] = spin
                    entropy[ii, jj] = res["entropy"][ii, jj, kk]
                    plaq[ii, jj] = res["plaq"][ii, jj, kk]
                    break
print("===========================")
for ii, g in enumerate(vals["g"]):
    for jj, m in enumerate(vals["m"]):
        if convergence[ii, jj] == 0:
            print(g, m)
            convergence[ii, jj] = 30
            entropy[ii, jj] = res["entropy"][ii, jj, -1]
            plaq[ii, jj] = res["plaq"][ii, jj, -1]
# %%
fig, ax = plt.subplots()
ax.plot(1 / vals["g"], convergence[:,], "-o")
ax.set(xscale="log", yscale="log")
ax.plot(1 / vals["g"], np.sqrt(2) / vals["g"])
# %%
fig, ax = plt.subplots()
img = ax.imshow(
    np.transpose(convergence),
    origin="lower",
    cmap="magma",
    extent=[-2, 1, -2, 1],
)
ax.set_ylabel(r"$m$")
ax.set_xlabel(r"$g$")
ax.set(yticks=[-2, -1, 0, 1], xticks=[-2, -1, 0, 1])
ax.xaxis.set_major_formatter(fake_log)
ax.yaxis.set_major_formatter(fake_log)

cb = fig.colorbar(
    img,
    ax=ax,
    aspect=20,
    location="right",
    orientation="vertical",
    pad=0.01,
)

fig, ax = plt.subplots()
img = ax.imshow(
    np.transpose(entropy),
    origin="lower",
    cmap="magma",
    extent=[-2, 1, -2, 1],
)
ax.set_ylabel(r"$m$")
ax.set_xlabel(r"$g$")
ax.set(yticks=[-2, -1, 0, 1], xticks=[-2, -1, 0, 1])
ax.xaxis.set_major_formatter(fake_log)
ax.yaxis.set_major_formatter(fake_log)

cb = fig.colorbar(
    img,
    ax=ax,
    aspect=20,
    location="right",
    orientation="vertical",
    pad=0.01,
)


fig, ax = plt.subplots()
img = ax.imshow(
    np.transpose(plaq),
    origin="lower",
    cmap="magma",
    extent=[-2, 1, -2, 1],
)
ax.set_ylabel(r"$m$")
ax.set_xlabel(r"$g$")
ax.set(yticks=[-2, -1, 0, 1], xticks=[-2, -1, 0, 1])
ax.xaxis.set_major_formatter(fake_log)
ax.yaxis.set_major_formatter(fake_log)

cb = fig.colorbar(
    img,
    ax=ax,
    aspect=20,
    location="right",
    orientation="vertical",
    pad=0.01,
)

save_dictionary(res, f"QED_conv.pkl")
# %%
config_filename = "Z2_FermiHubbard/U_potential"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["has_obc", "U"])

res = {}
# List of local observables
lvals = get_sim(ugrid[0][0]).par["lvals"]
local_obs = [f"n_{s}{d}" for d in "xyz"[: len(lvals)] for s in "mp"]
local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
local_obs += ["X_Cross"]
for obs in local_obs:
    res[obs] = np.zeros((vals["has_obc"].shape[0], vals["U"].shape[0]))

res["energy"] = np.zeros((vals["has_obc"].shape[0], vals["U"].shape[0]))

for ii, has_obc in enumerate(vals["has_obc"]):
    for jj, U in enumerate(vals["U"]):
        res["energy"][ii, jj] = get_sim(ugrid[ii][jj]).res["energies"][0] / (
            prod(lvals)
        )
        for obs in local_obs:
            res[obs][ii, jj] = np.mean(get_sim(ugrid[ii][jj]).res[obs])

fig = plt.figure()
plt.ylabel(r"X_cross")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["X_Cross"][ii, :], "-o", label=BC_label)
plt.legend()

fig = plt.figure()
plt.ylabel(r"N")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["N_pair"][ii, :], "-o", label=f"pair ({BC_label})")
    plt.plot(vals["U"], res["N_single"][ii, :], "-o", label=f"single ({BC_label})")
plt.legend()

fig = plt.figure()
plt.ylabel(r"Energy Density")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["energy"][ii, :], "-o", label=BC_label)
plt.legend(loc="lower right")

# %%
# ========================================================================
# ISING MODEL 1D ENERGY GAPS
# ========================================================================
config_filename = "Ising/Ising1D"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["lvals", "h"])
res = {
    "th_gap": np.zeros((vals["lvals"].shape[0], vals["h"].shape[0])),
    "true_gap": np.zeros((vals["lvals"].shape[0], vals["h"].shape[0])),
}
for ii, lvals in enumerate(vals["lvals"]):
    for jj, h in enumerate(vals["h"]):
        for obs in res.keys():
            res[obs][ii, jj] = get_sim(ugrid[ii][jj]).res[obs]

res["abs_distance"] = np.zeros((vals["lvals"].shape[0], vals["h"].shape[0]))
res["rel_distance"] = np.zeros((vals["lvals"].shape[0], vals["h"].shape[0]))
for ii, lvals in enumerate(vals["lvals"]):
    for jj, h in enumerate(vals["h"]):
        res["abs_distance"][ii, jj] = np.abs(
            res["true_gap"][ii, jj] - res["th_gap"][ii, jj]
        )
        res["rel_distance"][ii, jj] = (
            res["abs_distance"][ii, jj] / res["true_gap"][ii, jj]
        )
for obs in ["Sz", "Sx"]:
    res[obs] = np.zeros((vals["lvals"].shape[0], vals["h"].shape[0]))
    for ii, lvals in enumerate(vals["lvals"]):
        for jj, h in enumerate(vals["h"]):
            res[obs][ii, jj] = np.mean(get_sim(ugrid[ii][jj]).res[obs])

fig = plt.figure()
plt.ylabel(r"True_Gap")
plt.xlabel(r"h")
plt.xscale("log")
plt.yscale("log")
plt.grid()
for ii, lvals in enumerate(vals["lvals"]):
    plt.plot(vals["h"][:], res["true_gap"][ii, :], "-o", label=f"L={lvals}")
plt.legend()
plt.savefig("True_Gap_log.pdf")


fig = plt.figure()
plt.ylabel(r"Abs difference (true gap - th gap)")
plt.xlabel(r"h")
plt.xscale("log")
plt.grid()
for ii, lvals in enumerate(vals["lvals"]):
    plt.plot(vals["h"], res["abs_distance"][ii, :], "-o", label=f"L={lvals}")
plt.legend()
plt.savefig("abs_diff.pdf")

fig = plt.figure()
plt.ylabel(r"Rel difference (true gap - th gap)/(true gap)")
plt.xlabel(r"h")
plt.xscale("log")
plt.yscale("log")
plt.grid()
for ii, lvals in enumerate(vals["lvals"]):
    plt.plot(vals["h"][4:], res["rel_distance"][ii, 4:], "-o", label=f"L={lvals}")
plt.legend()
plt.savefig("rel_diff.pdf")

fig = plt.figure()
plt.ylabel(r"Sz")
plt.xlabel(r"h")
plt.xscale("log")
plt.grid()
for ii, lvals in enumerate(vals["lvals"]):
    plt.plot(vals["h"], res["Sz"][ii, :], "-o", label=f"L={lvals}")
plt.legend()
plt.savefig("Sz.pdf")


# %%
def LGT_obs_list(model, pure=None, has_obc=True):
    obs_list = [
        "energy",
        "entropy",
        "E_square",
        "plaq",
    ]
    if model == "SU2":
        obs_list += ["delta_E_square", "delta_plaq"]
        if not pure:
            obs_list += [
                "n_single_even",
                "n_single_odd",
                "n_pair_even",
                "n_pair_odd",
                "n_tot_even",
                "n_tot_odd",
                "delta_n_single_even",
                "delta_n_single_odd",
                "delta_n_pair_even",
                "delta_n_pair_odd",
                "delta_n_tot_even",
                "delta_n_tot_odd",
            ]
        if not has_obc:
            obs_list += ["py_sector", "px_sector"]
    else:
        obs_list += ["N"]
    return obs_list


# ========================================================================
# QED ENTANGLEMENT SCALING
# ========================================================================
config_filename = "QED/entanglement"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["spin", "g"])
obs_list = LGT_obs_list(model="QED", pure=False, has_obc=False)
res = {"g": vals["g"], "spin": vals["spin"]}
res["entropy"] = np.zeros((res["spin"].shape[0], res["g"].shape[0]))
for ii, s in enumerate(res["spin"]):
    for jj, g in enumerate(res["g"]):
        res["entropy"][ii][jj] = get_sim(ugrid[ii][jj]).res["entropy"]
fig = plt.figure()
plt.ylabel(r"Entanglement entropy")
plt.xlabel(r"g")
plt.xscale("log")
plt.grid()
for ii, s in enumerate(res["spin"]):
    plt.plot(res["g"], res["entropy"][ii, :], "-o", label=f"s={s}")
plt.legend()
save_dictionary(res, "saved_dicts/QED_entanglement.pkl")
# %%
# ========================================================================
# QED SINGULAR VALUES
# ========================================================================
config_filename = "QED/DM_scaling_PBC"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = LGT_obs_list(model="QED", pure=False, has_obc=False)
res = {"g": vals["g"]}
res["rho0"] = []
res["rho1"] = []
for ii, g in enumerate(res["g"]):
    res["rho0"].append(get_sim(ugrid[ii]).res["rho_eigvals"][0][::-1])
    res["rho1"].append(get_sim(ugrid[ii]).res["rho_eigvals"][1][::-1])
fig = plt.figure()
plt.ylabel(r"Value")
plt.yscale("log")
plt.xlabel(r"Singular Values")
plt.grid()
for ii, g in enumerate(res["g"]):
    plt.plot(np.arange(35), res["rho0"][ii], "-", label=f"g={format(g,'.3f')}")
plt.legend()
fig = plt.figure()
plt.ylabel(r"Value")
plt.yscale("log")
plt.xlabel(r"Singular Values")
plt.grid()
for ii, g in enumerate(res["g"]):
    plt.plot(np.arange(35), res["rho1"][ii], "-", label=f"g={format(g,'.3f')}")
plt.legend()
save_dictionary(res, "saved_dicts/QED_singular_values.pkl")

# %%
# ========================================================================
# QED Energy Gap convergence with different Parallel Transporters
# ========================================================================
U_definitions = ["spin", "ladder"]
for U in U_definitions:
    config_filename = f"QED/U_{U}"
    match = SimsQuery(group_glob=config_filename)
    ugrid, vals = uids_grid(match.uids, ["g", "spin"])
    res = {"g": vals["g"], "spin": vals["spin"]}
    res_shape = (res["g"].shape[0], res["spin"].shape[0])
    for obs in ["DeltaE", "E0", "E1", "B0", "B1", "DeltaB"]:
        res[obs] = np.zeros(res_shape)
    for ii, g in enumerate(res["g"]):
        for jj, spin in enumerate(res["spin"]):
            res["E0"][ii, jj] = get_sim(ugrid[ii, jj]).res["energy"][0]
            res["E1"][ii, jj] = get_sim(ugrid[ii, jj]).res["energy"][1]
            res["B0"][ii, jj] = get_sim(ugrid[ii, jj]).res["plaq"][0]
            res["B1"][ii, jj] = get_sim(ugrid[ii, jj]).res["plaq"][1]
            res["DeltaE"][ii, jj] = get_sim(ugrid[ii, jj]).res["DeltaE"]
            res["DeltaB"][ii, jj] = np.abs(res["B1"][ii, jj] - res["B0"][ii, jj])

    for ii, g in enumerate(res["g"]):
        beta = 1 / (g**2)
        fig = plt.figure()
        plt.ylabel(r"|Delta E|")
        plt.xlabel(r"s")
        plt.yscale("log")
        plt.grid()
        plt.plot(res["spin"][:], res["DeltaE"][ii, :], "-o", label=f"beta={beta}")
        plt.legend()
    save_dictionary(res, f"saved_dicts/QED_U_{U}.pkl")

# %%
# ========================================================================
# SU(2) SIMULATIONS PURE FLUCTUATIONS
# ========================================================================
config_filename = "SU2/pure/fluctuations"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = LGT_obs_list(model="SU2", pure=True, has_obc=True)
res = {"g": vals["g"]}
for obs in obs_list:
    res[obs] = []
    for ii in range(len(res["g"])):
        res[obs].append(get_sim(ugrid[ii]).res[obs])
    res[obs] = np.asarray(res[obs])

fig, ax = plt.subplots()
ax.plot(res["g"], res["E_square"], "-o", label=f"E2")
ax.plot(res["g"], res["delta_E_square"], "-o", label=f"Delta")
ax.set(xscale="log")
ax2 = ax.twinx()
ax2.plot(res["g"], -res["plaq"] + max(res["plaq"]), "-^", label=f"B2")
ax2.plot(res["g"], res["delta_plaq"], "-*", label=f"DeltaB")
ax.legend()
ax2.legend()
ax.grid()
save_dictionary(res, "saved_dicts/SU2_pure_fluctuations.pkl")
# %%
# ========================================================================
# SU(2) SIMULATIONS PURE TOPOLOGY
# ========================================================================
config_filename = "SU2/pure/topology"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = LGT_obs_list(model="SU2", pure=True, has_obc=False)
res = {"g": vals["g"]}

for obs in ["energy", "py_sector", "px_sector"]:
    res[obs] = np.zeros((vals["g"].shape[0], 5))
    for ii in range(len(res["g"])):
        for n in range(5):
            res[obs][ii][n] = get_sim(ugrid[ii]).res[obs][n]
fig = plt.figure()
for n in range(1, 5):
    plt.plot(
        vals["g"],
        res["energy"][:, n] - res["energy"][:, 0],
        "-o",
        label=f"{format(res['px_sector'][0, n],'.5f')}",
    )
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.ylabel("energy")
save_dictionary(res, "saved_dicts/SU2_pure_topology.pkl")
# %%
# ========================================================================
# SU(2) FULL TOPOLOGY 1
# ========================================================================
config_filename = "SU2/full/topology1"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=True)
res = {"g": vals["g"], "m": vals["m"]}
for obs in ["energy", "py_sector", "px_sector"]:
    res[obs] = np.zeros((res["g"].shape[0], res["m"].shape[0]))
    for ii, g in enumerate(res["g"]):
        for jj, m in enumerate(res["m"]):
            res[obs][ii][jj] = get_sim(ugrid[ii][jj]).res[obs]
fig = plt.figure()
for jj, m in enumerate(res["m"]):
    plt.plot(vals["g"], 1 - res["py_sector"][:, jj], "-o", label=f"m={format(m,'.3f')}")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.ylabel("1-py_sector")
save_dictionary(res, "saved_dicts/SU2_full_topology1.pkl")
# %%
# ========================================================================
# SU(2) FULL TOPOLOGY 2
# ========================================================================
config_filename = "SU2/full/topology2"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=True)
res = {"g": vals["g"], "m": vals["m"]}
for obs in ["energy", "py_sector", "px_sector"]:
    res[obs] = np.zeros((res["g"].shape[0], res["m"].shape[0]))
    for ii, g in enumerate(res["g"]):
        for jj, m in enumerate(res["m"]):
            res[obs][ii][jj] = get_sim(ugrid[ii][jj]).res[obs]
fig = plt.figure()
for ii, g in enumerate(res["g"]):
    plt.plot(vals["m"], 1 - res["py_sector"][ii, :], "-o", label=f"g={format(g,'.3f')}")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.ylabel("1-py_sector")
save_dictionary(res, "saved_dicts/SU2_full_topology2.pkl")
# %%
# ========================================================================
# SU(2) PHASE DIAGRAM
# ========================================================================
config_filename = "SU2/full/phase_diagram"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=False)
res = {"g": vals["g"], "m": vals["m"]}
for obs in obs_list:
    res[obs] = np.zeros((res["g"].shape[0], res["m"].shape[0]))
    for ii, g in enumerate(res["g"]):
        for jj, m in enumerate(res["m"]):
            res[obs][ii][jj] = get_sim(ugrid[ii][jj]).res[obs]

fig, axs = plt.subplots(
    3,
    1,
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
obs = ["E_square", "rho", "spin"]
for ii, ax in enumerate(axs.flat):
    # IMSHOW
    img = ax.imshow(
        np.transpose(res[obs[ii]]),
        origin="lower",
        cmap="magma",
        extent=[-2, 2, -3, 1],
    )
    ax.set_ylabel(r"m")
    axs[2].set_xlabel(r"g2")
    ax.set(xticks=[-2, -1, 0, 1, 2], yticks=[-3, -2, -1, 0, 1])
    ax.xaxis.set_major_formatter(fake_log)
    ax.yaxis.set_major_formatter(fake_log)

    cb = fig.colorbar(
        img,
        ax=ax,
        aspect=20,
        location="right",
        orientation="vertical",
        pad=0.01,
        label=obs[ii],
    )
save_dictionary(res, "saved_dicts/SU2_full_phase_diagram.pkl")
# %%
# ========================================================================
# SU(2) FULL THEORY CHARGE vs DENSITY
# ========================================================================
config_filename = "SU2/full/charge_vs_density"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=True)
res = {"g": vals["g"], "m": vals["m"]}
for obs in ["n_tot_even", "n_tot_odd"]:
    res[obs] = np.zeros((res["g"].shape[0], res["m"].shape[0]))
    for ii, g in enumerate(res["g"]):
        for jj, m in enumerate(res["m"]):
            res[obs][ii][jj] = get_sim(ugrid[ii][jj]).res[obs]
fig = plt.figure()
for ii, m in enumerate(res["m"]):
    plt.plot(
        vals["g"],
        2 + res["n_tot_even"][:, ii] - res["n_tot_odd"][:, ii],
        "-o",
        label=f"g={format(m,'.3f')}",
    )
plt.xscale("log")
plt.legend()
plt.ylabel("rho")
save_dictionary(res, "saved_dicts/charge_vs_density.pkl")
# %%
# ========================================================================
# SU(2) FULL THEORY TTN COMPARISON
# ========================================================================
config_filename = "SU2/full/TTN_comparison"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=True)
res = {"g": vals["g"]}
for obs in ["energy", "n_tot_even", "n_tot_odd", "E_square"]:
    res[obs] = np.zeros(res["g"].shape[0])
    for ii, g in enumerate(res["g"]):
        res[obs][ii] = get_sim(ugrid[ii]).res[obs]
fig = plt.figure()
plt.plot(vals["g"], 2 + res["n_tot_even"][:] - res["n_tot_odd"][:], "-o")
plt.xscale("log")
plt.legend()
plt.ylabel("rho")
save_dictionary(res, "saved_dicts/TTN_comparison.pkl")
# %%
# ========================================================================
# SU(2) FULL THEORY ENERGY GAPS
# ========================================================================
config_filename = "SU2/full/energy_gaps"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["DeltaN", "g", "k"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=True)
res = {"DeltaN": vals["DeltaN"], "g": vals["g"], "k": vals["k"]}
res_shape = (res["DeltaN"].shape[0], res["g"].shape[0], res["k"].shape[0])
res["energy"] = np.zeros(res_shape)
res["m"] = np.zeros(res_shape)
for ii, DeltaN in enumerate(res["DeltaN"]):
    for jj, g in enumerate(res["g"]):
        for kk, k in enumerate(res["k"]):
            res["m"][ii, jj, kk] = get_sim(ugrid[ii, jj, kk]).res["m"]
            res["energy"][ii, jj, kk] = get_sim(ugrid[ii, jj, kk]).res["energy"][0]

fig = plt.figure()
for kk, k in enumerate(res["k"]):
    plt.plot(
        vals["g"] ** 2,
        res["energy"][1, :, kk] - res["energy"][0, :, kk],
        "--o",
        label=f"k={k}, TOT",
    )


plt.xscale("log")
plt.yscale("log")
plt.xlabel("g2")
plt.legend()
plt.ylabel("DEltaE")

fig = plt.figure()
for kk, k in enumerate(res["k"]):
    plt.plot(
        vals["g"] ** 2,
        res["energy"][1, :, kk] - res["energy"][0, :, kk] - 0.5 * res["m"][1, :, kk],
        "-^",
        label=f"k={k} RES",
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("g2")
plt.legend()
plt.ylabel("DEltaE_res")
save_dictionary(res, "saved_dicts/SU2_energy_gap.pkl")

# %%
# ========================================================================
# SU(2) FULL THEORY ENERGY GAPS
# ========================================================================
config_filename = "SU2/full/energy_gaps"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["DeltaN", "m", "k"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=False)
res = {"DeltaN": vals["DeltaN"], "m": vals["m"], "k": vals["k"]}
res_shape = (res["DeltaN"].shape[0], res["m"].shape[0], res["k"].shape[0])
res["energy"] = np.zeros(res_shape)
res["g"] = np.zeros(res_shape)
for ii, DeltaN in enumerate(res["DeltaN"]):
    for jj, m in enumerate(res["m"]):
        for kk, k in enumerate(res["k"]):
            res["g"][ii, jj, kk] = get_sim(ugrid[ii, jj, kk]).res["g"]
            res["energy"][ii, jj, kk] = get_sim(ugrid[ii, jj, kk]).res["energy"][0]

fig = plt.figure()
for kk, k in enumerate(res["k"]):
    plt.plot(
        res["m"],
        res["energy"][1, :, kk] - res["energy"][0, :, kk],
        "--o",
        label=f"k={k}",
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("m")
plt.legend()
plt.ylabel("DEltaE")

fig = plt.figure()
for kk, k in enumerate(res["k"]):
    plt.plot(
        vals["m"],
        res["energy"][1, :, kk] - res["energy"][0, :, kk] - 0.5 * res["m"],
        "-^",
        label=f"k={k} RES",
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("g2")
plt.legend()
plt.ylabel("DEltaE_res")
save_dictionary(res, "SU2_energy_gap_new.pkl")
