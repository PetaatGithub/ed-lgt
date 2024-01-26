from ed_lgt.models import IsingModel
from simsio import run_sim

with run_sim() as sim:
    sim.par["coeffs"] = {"J": sim.par["J"], "h": sim.par["h"]}
    model = IsingModel(sim.par)
    # GET OPERATORS
    model.get_operators(sparse=False)
    # GET SYMMETRY SECTOR
    sym_sector = sim.par["sym_sector"]
    if sym_sector is not None:
        model.get_abelian_symmetry_sector(["Sz"], [sym_sector], sym_type="P")
    # BUILD AND DIAGONALIZE HAMILTONIAN
    model.build_Hamiltonian()
    model.diagonalize_Hamiltonian()
    # LIST OF LOCAL OBSERVABLES
    local_obs = ["Sx", "Sz"]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [
        ["Sz", "Sz"],
        ["Sx", "Sm"],
        ["Sx", "Sp"],
        ["Sp", "Sx"],
        ["Sm", "Sx"],
    ]
    # DEFINE OBSERVABLES
    model.get_observables(local_obs, twobody_obs)
    for ii in range(model.n_eigs):
        # PRINT ENERGY
        model.H.print_energy(ii)
        # PRINT STATE CONFIGURATIONS
        model.H.Npsi[ii].get_state_configurations()
        # MEASURE OBSERVABLES
        model.measure_observables(ii)
        if ii == 0:
            # MEASURE THE GAP
            model.get_energy_gap()
            for measure in model.res.keys():
                sim.res[measure] = model.res[measure]
