import numpy as np
from ed_lgt.models import SU2_Model
from ed_lgt.operators import SU2_Hamiltonian_couplings
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    start_time = perf_counter()
    model = SU2_Model(**sim.par["model"])
    # GLOBAL SYMMETRIES
    if model.pure_theory:
        global_ops = None
        global_sectors = None
    else:
        global_ops = [model.ops[op] for op in sim.par["symmetries"]["sym_ops"]]
        global_sectors = sim.par["symmetries"]["sym_sectors"]
    # LINK SYMMETRIES
    link_ops = [
        [model.ops[f"T2_p{d}"], -model.ops[f"T2_m{d}"]] for d in model.directions
    ]
    link_sectors = [0 for _ in model.directions]
    # GET SYMMETRY SECTOR
    model.get_abelian_symmetry_sector(
        global_ops=global_ops,
        global_sectors=global_sectors,
        link_ops=link_ops,
        link_sectors=link_sectors,
    )
    # DEFAUL PARAMS
    model.default_params()
    # BUILD AND DIAGONALIZE HAMILTONIAN
    coeffs = SU2_Hamiltonian_couplings(
        model.dim, model.pure_theory, sim.par["g"], sim.par["m"]
    )
    model.build_Hamiltonian(coeffs)
    model.diagonalize_Hamiltonian(n_eigs=sim.par["hamiltonian"]["n_eigs"])
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"T2_{s}{d}" for d in model.directions for s in "mp"]
    local_obs += ["E_square"]
    if not model.pure_theory:
        local_obs += [f"N_{label}" for label in ["r", "g", "tot", "single", "pair"]]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [[f"P_p{d}", f"P_m{d}"] for d in model.directions]
    twobody_axes = [d for d in model.directions]
    # LIST OF PLAQUETTE OPERATORS
    plaquette_obs = []
    # DEFINE OBSERVABLES
    model.get_observables(
        local_obs, twobody_obs, plaquette_obs, twobody_axes=twobody_axes
    )
    # MEASUREMENTS
    sim.res["entropy"] = np.zeros(model.n_eigs, dtype=float)
    sim.res["energy"] = model.H.Nenergies
    sim.res["overlap_V"] = np.zeros(model.n_eigs, dtype=float)
    sim.res["overlap_PV"] = np.zeros(model.n_eigs, dtype=float)
    sim.res["overlap_M"] = np.zeros(model.n_eigs, dtype=float)
    sim.res["indices"] = []
    # Vacuum and Bare vacuum
    vacuum = np.array(
        [
            (
                0
                if i % 2 == 0
                else 4 if (i < (model.lvals[0] - 1) or not model.has_obc[0]) else 2
            )
            for i in range(model.lvals[0])
        ]
    )
    p_vacuum = np.array(
        [
            (
                1
                if i % 2 == 0
                else 5 if (i < (model.lvals[0] - 1) or not model.has_obc[0]) else 1
            )
            for i in range(model.lvals[0])
        ]
    )
    m_vacuum = np.array([1, 3, 0, 2, 3, 4, 2, 1])
    row_ind1 = np.where((model.sector_configs == vacuum).all(axis=1))[0]
    row_ind2 = np.where((model.sector_configs == p_vacuum).all(axis=1))[0]
    row_ind3 = np.where((model.sector_configs == m_vacuum).all(axis=1))[0]

    for ii in range(model.n_eigs):
        # PRINT ENERGY
        model.H.print_energy(ii)
        # ENTROPY
        sim.res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
            list(np.arange(0, int(model.lvals[0] / 2), 1)),
            sector_configs=model.sector_configs,
        )
        sim.res["overlap_V"][ii] = np.abs(model.H.Npsi[ii].psi[row_ind1]) ** 2
        sim.res["overlap_M"][ii] = np.abs(model.H.Npsi[ii].psi[row_ind3]) ** 2
        sim.res["overlap_PV"][ii] = np.abs(model.H.Npsi[ii].psi[row_ind2]) ** 2
        if sim.res["entropy"][ii] < 1e-1:
            # STATE CONFIGURATIONS
            model.H.Npsi[ii].get_state_configurations(
                threshold=1e-2, sector_configs=model.sector_configs
            )
        # MEASURE OBSERVABLES
        # model.measure_observables(ii)
        # CHECK LINK SYMMETRIES
        # model.check_symmetries()
        """
        if ii == 0:
            # SAVE RESULTS
            for measure in model.res.keys():
                sim.res[measure] = model.res[measure]
        """
    # Define the intervals
    intervals = [(-40, -30), (-25, -15), (-10, 0), (0, 20)]
    # Prepare to collect the results
    max_entropy_energies = []
    max_entropy_indices = []
    # Process each interval
    for lower_bound, upper_bound in intervals:
        # Find indices where energy falls within the interval
        valid_indices = np.where(
            (sim.res["energy"] >= lower_bound) & (sim.res["energy"] <= upper_bound)
        )[0]

        if valid_indices.size > 0:
            # Find the index of the maximum entropy within this range
            max_index = valid_indices[np.argmax(sim.res["overlap_V"][valid_indices])]
            # STATE CONFIGURATIONS
            model.H.Npsi[max_index].get_state_configurations(
                threshold=1e-2, sector_configs=model.sector_configs
            )
            sim.res["indices"].append(max_index)

    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
