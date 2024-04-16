import numpy as np
from numpy.linalg import eig
from scipy.sparse.linalg import norm
from itertools import product, combinations
from scipy.sparse import csr_matrix, diags, identity, kron
from ed_lgt.tools import (
    check_commutator as check_comm,
    anti_commutator as anti_comm,
    validate_parameters,
)
from .bose_fermi_operators import fermi_operators as Zn_matter_operators
from ed_lgt.modeling import qmb_operator as qmb_op, get_lattice_borders_labels

__all__ = [
    "Zn_rishon_operators",
    "Zn_dressed_site_operators",
    "Zn_magnetic_site_operators",
    "Zn_gauge_invariant_states",
    "Zn_gauge_invariant_ops",
]


def Zn_rishon_operators(n, pure_theory):
    """
    This function constructs the Operators (E,U) of a Zn Lattice Gauge Theory
    in the Electric Basis (where E is diagonal) and provides the corresponding
    Rishon Operators that are suitable for a dressed site description.
    NOTE: in the pure theory, n must be ODD. In the full theory, n must be EVEN

    Args:
        n (int): dimension of the gauge link Hilbert space

        pure_theory (bool): if True, it only provides gauge link operators.
            If False, it also provides matter field operators and requires n to be even.

    Returns:
        dict: dictionary with Zn operators
    """
    validate_parameters(int_list=[n], pure_theory=pure_theory)
    # Size of the gauge Hilbert space
    size = n
    shape = (size, size)
    if pure_theory:
        if size % 2 == 0:
            raise ValueError(f"Zn pure dressed sites work only n odd")
    else:
        if size % 2 != 0:
            raise ValueError(f"Matter dressed sites work only for Zn with n even")
    # DICTIONARY OF OPERATORS
    ops = {}
    # PARALLEL TRANSPORTER
    U_diag = [np.ones(size - 1), np.ones(1)]
    ops["U"] = diags(U_diag, [+1, -size + 1], shape)
    # IDENTITY OPERATOR
    ops["IDz"] = identity(size)
    # ELECTRIC FIELD OPERATORS
    ops["n"] = diags(np.arange(size)[::-1], 0, shape)
    ops["E"] = ops["n"] - 0.5 * (size - 1) * identity(size)
    ops["E_square"] = ops["E"] ** 2
    # RISHON OPERATORS
    if pure_theory:
        # PARITY OPERATOR
        ops["P"] = identity(size)
    else:
        # PARITY OPERATOR
        ops["P"] = diags([(-1) ** ii for ii in range(size)], 0, shape)
    # RISHON OPERATORS
    ops["Zp"] = ops["P"] * ops["U"]
    ops["Zm"] = ops["U"]
    for s in "pm":
        ops[f"Z{s}_dag"] = ops[f"Z{s}"].transpose()
    # Useful operators for Corners
    ops["Zm_P"] = ops["Zm"] * ops["P"]
    ops["Zp_P"] = ops["Zp"] * ops["P"]
    ops["P_Zm_dag"] = ops["P"] * ops["Zm_dag"]
    ops["P_Zp_dag"] = ops["P"] * ops["Zp_dag"]
    if not pure_theory:
        # PERFORM CHECKS
        for s1, s2 in zip("pm", "mp"):
            # CHECK RISHON MODES TO BEHAVE LIKE FERMIONS
            # anticommute with parity
            a = anti_comm(ops[f"Z{s1}"], ops["P"])
            if norm(a) > 1e-15:
                print(a.todense())
                raise ValueError(f"Z{s1} must anticommute with Parity")
            b = anti_comm(ops[f"Z{s1}"], ops[f"Z{s2}_dag"])
            if norm(b) > 1e-15:
                print(b.todense())
                raise ValueError(f"Z{s1} and Z{s2}_dag must anticommute")
    return ops


def truncated_fourier_transform(U, n):
    eigs, F = eig(a=U.toarray())
    return eigs[-n:], F[:, -n:]


def Zn_magnetic_site_operators(n, truncation):
    validate_parameters(int_list=[n, truncation])
    # Get the Rishon operators according to the chosen n representation s
    in_ops = Zn_rishon_operators(n, True)
    # Diagonalize the parallel transporter U, and select a truncated basis F of angles
    eigs, F = truncated_fourier_transform(in_ops["U"], truncation)
    ops = {
        "T": csr_matrix(2 * np.diag(eigs)),
        "E": csr_matrix(F.conj().transpose() @ in_ops["E"] @ F),
    }
    return ops


def Zn_dressed_site_operators(n, pure_theory, lattice_dim):
    """
    This function generates the dressed-site operators of the 2D Zn Hamiltonian
    (pure or with matter fields) for any possible value n of Zn
    (the larger n the larger the gauge link Hilbert space) in the Electric Basis

    Args:
        n (int): dimension of the gauge link Hilbert space

        pure_theory (bool, optional): If true, the dressed site includes matter fields. Defaults to False.

    Returns:
        dict: dictionary with all the operators of the QED (pure or full) Hamiltonian
    """
    validate_parameters(int_list=[n], pure_theory=pure_theory, lattice_dim=lattice_dim)
    # Get the Rishon operators according to the chosen n representation s
    in_ops = Zn_rishon_operators(n, pure_theory)
    # Dictionary for operators
    ops = {}
    if lattice_dim == 2:
        # Electric Operators
        for op in ["n", "E"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "IDz", "IDz", "IDz"])
            ops[f"{op}_my"] = qmb_op(in_ops, ["IDz", op, "IDz", "IDz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["IDz", "IDz", op, "IDz"])
            ops[f"{op}_py"] = qmb_op(in_ops, ["IDz", "IDz", "IDz", op])
        # Corner Operators: in this case the rishons are bosons: no need of parities
        ops["C_px,py"] = qmb_op(in_ops, ["IDz", "IDz", "Zm_P", "Zp_dag"])  # -1
        ops["C_py,mx"] = qmb_op(in_ops, ["P_Zp_dag", "P", "P", "Zm"])
        ops["C_mx,my"] = qmb_op(in_ops, ["Zm_P", "Zp_dag", "IDz", "IDz"])
        ops["C_my,px"] = qmb_op(in_ops, ["IDz", "Zm_P", "Zp_dag", "IDz"])
        if not pure_theory:
            # Acquire also matter field operators
            in_ops |= Zn_matter_operators(has_spin=False)
            if not pure_theory:
                # Update Electric and Corner operators
                for op in ops.keys():
                    ops[op] = kron(in_ops["ID_psi"], ops[op])
            # Hopping operators
            ops["Q_mx_dag"] = qmb_op(in_ops, ["psi_dag", "Zm", "IDz", "IDz", "IDz"])
            ops["Q_my_dag"] = qmb_op(in_ops, ["psi_dag", "P", "Zm", "IDz", "IDz"])
            ops["Q_px_dag"] = qmb_op(in_ops, ["psi_dag", "P", "P", "Zp", "IDz"])
            ops["Q_py_dag"] = qmb_op(in_ops, ["psi_dag", "P", "P", "P", "Zp"])
            # Add dagger operators
            Qs = {}
            for op in ops:
                dag_op = op.replace("_dag", "")
                Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
            ops |= Qs
            # Psi Number operators
            ops["N"] = qmb_op(in_ops, ["N", "IDz", "IDz", "IDz", "IDz"])
        # Sum all the E_square operators with coefficient 1/2
        ops["E_square"] = 0
        for s in ["mx", "my", "px", "py"]:
            ops["E_square"] += 0.5 * ops[f"E_{s}"] ** 2
        # Check that corner operators commute
        corner_list = ["C_mx,my", "C_py,mx", "C_my,px", "C_px,py"]
        for C1, C2 in combinations(corner_list, 2):
            check_comm(ops[C1], ops[C2])
    return ops


def Zn_gauge_invariant_states(n, pure_theory, lattice_dim):
    """
    This function generates the gauge invariant basis of a Zn LGT
    in a d-dimensional lattice where gauge (and matter) degrees of
    freedom are merged in a compact-site notation by exploiting
    a rishon-based formalism.

    NOTE: The function provides also a restricted basis for sites
    on the borderd of the lattice where not all the configurations
    are allowed (the external rishons/gauge fields do not contribute)

    NOTE: for the moment, it works only for the pure case

    Args:
        n (scalar, int): size of the Hilbert space of the Zn Gauge field

        pure_theory (bool,optional): if True, the theory does not involve matter fields

        lattice_dim (int, optional): number of spatial dimensions. Defaults to 2.

    Returns:
        (dict, dict): dictionaries with the basis and the states
    """
    validate_parameters(int_list=[n], pure_theory=pure_theory, lattice_dim=lattice_dim)
    rishon_size = n
    single_rishon_configs = np.arange(rishon_size)
    # List of borders/corners of the lattice
    borders = get_lattice_borders_labels(lattice_dim)
    # List of configurations for each element of the dressed site
    dressed_site_config_list = [single_rishon_configs for i in range(2 * lattice_dim)]
    # Distinction between pure and full theory
    if pure_theory:
        core_labels = ["site"]
        parity = [1]
    else:
        core_labels = ["even", "odd"]
        parity = [1, -1]
        dressed_site_config_list.insert(0, np.arange(2))
    # Define useful quantities
    gauge_states = {}
    row = {}
    col_counter = {}
    for ii, main_label in enumerate(core_labels):
        row_counter = -1
        gauge_states[main_label] = []
        row[main_label] = []
        col_counter[main_label] = -1
        for label in borders:
            gauge_states[f"{main_label}_{label}"] = []
            row[f"{main_label}_{label}"] = []
            col_counter[f"{main_label}_{label}"] = -1
        # Look at all the possible configurations of gauge links and matter fields
        for config in product(*dressed_site_config_list):
            # Update row counter
            row_counter += 1
            # Define Gauss Law
            left = sum(config)
            right = lattice_dim * (rishon_size - 1) + 0.5 * (1 - parity[ii])
            # Check Gauss Law
            if (left - right) % n == 0:
                # FIX row and col of the site basis
                row[main_label].append(row_counter)
                col_counter[main_label] += 1
                # Save the gauge invariant state
                gauge_states[main_label].append(config)
                # Get the config labels
                label = Zn_border_configs(config, n, pure_theory)
                if label:
                    # save the config state also in the specific subset for the specif border
                    for ll in label:
                        gauge_states[f"{main_label}_{ll}"].append(config)
                        row[f"{main_label}_{ll}"].append(row_counter)
                        col_counter[f"{main_label}_{ll}"] += 1
    # Build the basis as a sparse matrix
    gauge_basis = {}
    for name in list(gauge_states.keys()):
        data = np.ones(col_counter[name] + 1, dtype=float)
        x = np.asarray(row[name])
        y = np.arange(col_counter[name] + 1)
        gauge_basis[name] = csr_matrix(
            (data, (x, y)), shape=(row_counter + 1, col_counter[name] + 1)
        )
        # Save the gauge states as a np.array
        gauge_states[name] = np.asarray(gauge_states[name])
    return gauge_basis, gauge_states


def Zn_gauge_invariant_ops(n, pure_theory, lattice_dim):
    in_ops = Zn_dressed_site_operators(n, pure_theory, lattice_dim)
    gauge_basis, _ = Zn_gauge_invariant_states(n, pure_theory, lattice_dim)
    E_ops = {}
    label = "site" if pure_theory else "even"
    for op in in_ops.keys():
        E_ops[op] = gauge_basis[label].transpose() @ in_ops[op] @ gauge_basis[label]
    return E_ops


def Zn_border_configs(config, n, pure_theory=True):
    """
    This function fixes the value of the electric field on
    lattices with open boundary conditions (has_obc=True).

    For integer spin representation, the offset of E is naturally
    the central value assumed by the rishon number.

    Args:
        config (list of ints): configuration of internal rishons in
        the single dressed site basis, ordered as follows:
        [n_matter, n_mx, n_my, n_px, n_py]

        spin (int): chosen spin representation for U(1)

        pure_theory (bool): True if the theory does not include matter

    Returns:
        list of strings: list of configs corresponding to a border/corner of the lattice
        with a fixed value of the electric field
    """
    if not isinstance(config, list) and not isinstance(config, tuple):
        raise TypeError(f"config should be a LIST, not a {type(config)}")
    validate_parameters(int_list=[n], pure_theory=pure_theory)
    off_set = 0
    label = []
    if not pure_theory:
        config = config[1:]
    if config[0] == off_set:
        label.append("mx")
    if config[1] == off_set:
        label.append("my")
    if config[2] == off_set:
        label.append("px")
    if config[3] == off_set:
        label.append("py")
    if (config[0] == off_set) and (config[1] == off_set):
        label.append("mx_my")
    if (config[0] == off_set) and (config[3] == off_set):
        label.append("mx_py")
    if (config[1] == off_set) and (config[2] == off_set):
        label.append("px_my")
    if (config[2] == off_set) and (config[3] == off_set):
        label.append("px_py")
    return label
