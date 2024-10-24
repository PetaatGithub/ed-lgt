import numpy as np
import logging
from numba import njit, prange
from ed_lgt.tools import get_time

logger = logging.getLogger(__name__)

__all__ = [
    "check_global_sym",
    "check_global_sym_sitebased",
    "global_abelian_sector",
    "global_sector_configs",
]


@njit
def check_global_sym(config, sym_op_diags, sym_sectors, sym_type_flag):
    """
    This function checks if a given QMB state configuration concurrently belongs to
    a specified GLOBAL abelian symmetry sector, which can be the intersection of an arbitrary
    large set of GLOBAL abelian symmetry sectors, each one generated by a different diagonal
    operator (which therefore commutes with the Hamiltonian).
    The type of allowed symmetries are U(1) and Zn

    NOTE: This function assumes all the QMB sites to have the same local dimension.

    Args:
        config (np.array of np.uint8): 1D array corresponding to a single QMB state configuration

        sym_op_diags (np.array of floats): 2D array: each row is the diagonal of one operator
            generating the symmetry sector. sym_op_diags=(num_generators, local_dimension)

        sym_sectors (np.array of floats): 1D array with sector values for each operator.
            NOTE: sym_sectors.shape[0] = sym_op_diags.shape[0]

        sym_type_flag (int): Flag indicating the symmetry type (0 for "U", 1 for "Zn").

    Returns:
        bool: True if the config belongs to the chosen sector, False otherwise
    """
    num_operators = sym_op_diags.shape[0]
    check = True
    # Run over all the number of symmetries
    for jj in range(num_operators):
        # Perform sum or product based on sym_type_flag
        if sym_type_flag == 0:
            # "U" for sum
            operation_result = np.sum(sym_op_diags[jj][config])
        else:
            # "Zn" for product
            operation_result = np.prod(sym_op_diags[jj][config])
        if not np.isclose(operation_result, sym_sectors[jj], atol=1e-10):
            check = False
            # Early exit on first failure
            return check
    return check


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
def global_abelian_sector(loc_dims, sym_op_diags, sym_sectors, sym_type):
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

    Returns:
        (np.array of ints, np.array of ints): 1D array of indices and 2D array of QMB state configurations
    """
    if not isinstance(sym_sectors, np.ndarray):
        sym_sectors = np.array(sym_sectors, dtype=float)
    # Acquire Sector dimension
    sector_dim = np.prod(loc_dims)
    logger.info(f"TOT DIM: {sector_dim}, 2^{round(np.log2(sector_dim),3)}")
    # Convert sym_type to a flag
    sym_type_flag = 0 if sym_type == "U" else 1
    # Select only the correct configs
    sector_configs = global_sector_configs(
        loc_dims, sym_op_diags, sym_sectors, sym_type_flag
    )
    sector_indices = np.ravel_multi_index(sector_configs.T, loc_dims)
    # Acquire dimension of the new sector
    sector_dim = len(sector_configs)
    logger.info(f"SEC DIM: {sector_dim}, 2^{round(np.log2(sector_dim),1)}")
    return sector_indices, sector_configs


@njit(parallel=True)
def global_sector_configs(loc_dims, glob_op_diags, glob_sectors, sym_type_flag):
    # =============================================================================
    # Get all the possible QMB state configurations
    # Total number of configs
    sector_dim = 1
    for dim in loc_dims:
        sector_dim *= dim
    # Len of each config
    num_dims = len(loc_dims)
    configs = np.zeros((sector_dim, num_dims), dtype=np.uint8)
    # Use an auxiliary array to mark valid configurations
    checks = np.zeros(sector_dim, dtype=np.bool_)
    # Iterate over all the possible configs
    for ii in prange(sector_dim):
        tmp = ii
        for dim_index in range(num_dims):
            divisor = (
                np.prod(loc_dims[dim_index + 1 :]) if dim_index + 1 < num_dims else 1
            )
            configs[ii, dim_index] = (tmp // divisor) % loc_dims[dim_index]
        # Check if the config satisfied the symmetries
        if check_global_sym_sitebased(
            configs[ii], glob_op_diags, glob_sectors, sym_type_flag
        ):
            checks[ii] = True
    # =============================================================================
    # Filter configs based on checks
    return configs[checks]
