import numpy as np
import logging
from numba import njit, prange
from ed_lgt.tools import get_time
from .generate_configs import get_state_configs

logger = logging.getLogger(__name__)

__all__ = [
    "check_global_sym_configs",
    "check_global_sym_configs_sitebased",
    "check_global_sym_sitebased",
    "global_abelian_sector",
]


@get_time
@njit(parallel=True)
def check_global_sym_configs(configs, sym_op_diags, sym_sectors, sym_type_flag):
    """
    This function selects the state configurations of a QMB state that belong to
    a specified GLOBAL abelian symmetry sector, which can be the intersection of an arbitrary
    large set of GLOBAL abelian symmetry sectors, each one generated by a different diagonal
    operator (which therefore commutes with the Hamiltonian).
    The type of allowed symmetries are U(1) and Zn

    NOTE: This function assumes all the QMB sites to have the same local dimension.

    Args:
        configs (np.array of np.uint8): 2D array with state configurations (each row) to check.

        sym_op_diags (np.array of floats): 2D array: each row is the diagonal of one operator
            generating the symmetry sector. sym_op_diags=(num_generators, local_dimension)

        sym_sectors (np.array of floats): 1D array with sector values for each operator.
            NOTE: sym_sectors.shape[0] = sym_op_diags.shape[0]

        sym_type_flag (int): Flag indicating the symmetry type (0 for "U", 1 for "Zn").

    Returns:
        np.array: shape=(configs.shape[0],) and dtype=bool
            Each bool entries label the corresponding state config (configs row)
            as belonging or not to the chosen symmetry sector.
    """
    num_configs = configs.shape[0]
    checks = np.zeros(num_configs, dtype=np.bool_)

    for ii in prange(num_configs):
        check = True
        # Run over all the number of symmetries
        for jj in range(sym_op_diags.shape[0]):
            # Perform sum or product based on sym_type_flag
            if sym_type_flag == 0:
                # "U" for sum
                operation_result = np.sum(sym_op_diags[jj][configs[ii]])
            else:
                # "Zn" for product
                operation_result = np.prod(sym_op_diags[jj][configs[ii]])
            if not np.isclose(operation_result, sym_sectors[jj], atol=1e-10):
                check = False
                break
        checks[ii] = check
    return checks


@get_time
@njit(parallel=True)
def check_global_sym_configs_sitebased(
    configs, sym_op_diags, sym_sectors, sym_type_flag
):
    """
    Thif function is an extension check_global_sym_configs to check configurations
    against symmetry sectors with site-specific operator diagonals, as in the case of
    Lattice Gauge Theories, where (dressed sites) have different local Hilbert spaces
    according to Boundary Conditions and lattice geometry.

    NOTE: Args are the same as before, but sym_op_diags has a different shape
    Namely, it is a 3D array of shape=(num_operators, n_sites, max(loc_dims))
    where each operator has its diagonal expressed in the proper basis of each lattice site.

    """
    num_configs = configs.shape[0]
    num_sites = configs.shape[1]
    num_operators = sym_op_diags.shape[0]
    checks = np.zeros(num_configs, dtype=np.bool_)

    for ii in prange(num_configs):
        check = True
        for jj in range(num_operators):
            # Initialize for sum (U) or product (Zn)
            operation_result = 0 if sym_type_flag == 0 else 1
            for kk in range(num_sites):
                # Actual dimension for the site
                op_value = sym_op_diags[jj, kk, configs[ii, kk]]
                # Distinguish between U(1) and Zn
                if sym_type_flag == 0:
                    # U for sum
                    operation_result += op_value
                else:
                    # Zn for product
                    operation_result *= op_value
            if not np.isclose(operation_result, sym_sectors[jj], atol=1e-10):
                check = False
                break
        checks[ii] = check
    return checks


@njit
def check_global_sym_sitebased(config, sym_op_diags, sym_sectors, sym_type_flag):
    """
    This function checks if a QMB state configuration belongs to a global abelian symmetry sector,
    which can be the intersection of an arbitrary large set of global abelian symmetry sectors,
    each one generated by a different diagonal operator (which therefore commutes with the Hamiltonian).
    The type of allowed symmetries are U(1) and Zn

    NOTE: In this case, we assume each site with a different Hilbert basis
    (as in Lattice Gauge Theories within the dressed site formalism).

    This function acts as check_global_sym_configs_sitebased but just on a single config

    Args:
        config (np.array of np.uint8): 1D array with the state of each lattice site.

        sym_op_diags (np.array of floats): 3D array of shape=(num_operators, n_sites, max(loc_dims))
            where each operator has its diagonal expressed in the proper basis of each lattice site.

        sym_sectors (np.array of floats): 1D array with sector values for each operator.
            NOTE: sym_sectors.shape[0] = sym_op_diags.shape[0]

        sym_type_flag (int): Flag indicating the symmetry type (0 for "U", 1 for "Zn").

    Returns:
        bool: True if the state belongs to the sector, False otherwise
    """
    num_sites = config.shape[0]
    num_operators = sym_op_diags.shape[0]

    check = True
    for jj in range(num_operators):
        # Initialize for sum (U) or product (Zn)
        operation_result = 0 if sym_type_flag == 0 else 1
        for kk in range(num_sites):
            # Actual dimension for the site
            op_value = sym_op_diags[jj, kk, config[kk]]
            # U for sum
            if sym_type_flag == 0:
                operation_result += op_value
            # Zn for product
            else:
                operation_result *= op_value
        if not np.isclose(operation_result, sym_sectors[jj], atol=1e-10):
            check = False
            break
    return check


@get_time
def global_abelian_sector(loc_dims, sym_op_diags, sym_sectors, sym_type, configs=None):
    """
    This function returns the QMB state configurations (and the corresponding 1D indices)
    that belongs to the intersection of multiple global symmetry sectors

    Args:
        loc_dims (np.array): 1D array of single-site local dimensions.
            For Exact Diagonlization (ED) purposes, each local dimension is always smaller that 2^{8}-1
            For this reason, loc_dims.dtype = np.uint8

        sym_op_diags (np.array of floats): Array with (diagonals of) operators generating the global
            symmetries of the model. If len(shape)=3, then it handles the case of different local
            Hilbert spaces, and, for each operator, its diagonal is evaluated on each site Hilbert basis

        sym_sectors (np.array of floats): 1D array with sector values for each operator.
            NOTE: sym_sectors.shape[0] = sym_op_diags.shape[0]

        sym_type (str): U for U(1) global symmetries, Z for Zn symmetries

        configs (np.array of np.uint8, optional):
            2D array with state configurations (each row) to check.
            If None, it is generated from loc_dims. Defaults to None.

    Returns:
        (np.array of ints, np.array of ints): 1D array of indices and 2D array of QMB state configurations
    """
    if configs is None:
        # Get QMB state configurations
        configs = get_state_configs(loc_dims)
    # Acquire Sector dimension
    sector_dim = len(configs)
    logger.info(f"TOT DIM: {sector_dim}, 2^{round(np.log2(sector_dim),1)}")
    # Convert sym_type to a flag
    sym_type_flag = 0 if sym_type == "U" else 1
    # Compute the check on the whole set of config
    if len(sym_op_diags.shape) == 2:
        checks = check_global_sym_configs(
            configs, sym_op_diags, sym_sectors, sym_type_flag
        )
    else:
        checks = check_global_sym_configs_sitebased(
            configs, sym_op_diags, sym_sectors, sym_type_flag
        )
    # Select only the correct configs
    sector_configs = configs[checks]
    sector_indices = np.ravel_multi_index(sector_configs.T, loc_dims)
    # Acquire dimension of the new sector
    sector_dim = len(sector_configs)
    logger.info(f"SEC DIM: {sector_dim}, 2^{round(np.log2(sector_dim),1)}")
    return (sector_indices, sector_configs)
