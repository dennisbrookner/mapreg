"""
Map registration

This file contains a python function and also a command-line utility
"""

import numpy as np
from tqdm import tqdm
import argparse

import reciprocalspaceship as rs
import gemmi
from skimage.transform import warp
from skimage.registration import optical_flow_ilk


def make_floatgrid(mtz, spacing, F, Phi, spacegroup=None):
    """
    Make a gemmi.FloatGrid object from an rs.DataSet object

    Parameters
    ----------
    mtz : rs.DataSet
        mtz data to be transformed into real space
    spacing : float
        Approximate voxel size desired (will be rounded as necessary to create integer grid dimensions)
    F : str, optional
        Column in mtz containing structure factor amplitudes to use for calculation, by default "2FOFCWT"
    Phi : str, optional
        Column in mtz containing phases to be used for calculation, by default "PH2FOFCWT"
    spacegroup : str, optional
        Spacegroup for the output FloatGrid. If None (default), FloatGrid will inherit the spacegroup of mtz.

    Returns
    -------
    float_grid : gemmi.FloatGrid
        Fourier transform of mtz, written out as a gemmi object containing a 3D voxel array
        and various other metadata and methods
    """

    # drop NAs in either of the specified columns
    # this has the secondary purpose of not silently modifying the input mtz
    new_mtz = mtz[~mtz[F].isnull()]
    new_mtz = new_mtz[~new_mtz[Phi].isnull()]

    new_mtz.hkl_to_asu(inplace=True)

    # compute desired grid size based on given spacing
    gridsize = []
    for dim in [new_mtz.cell.a, new_mtz.cell.b, new_mtz.cell.c]:
        gridsize.append(int(dim // spacing))

    # perform FFT using the desired amplitudes and phases
    new_mtz["Fcomplex"] = new_mtz.to_structurefactor(F, Phi)
    reciprocal_grid = new_mtz.to_reciprocal_grid("Fcomplex", grid_size=gridsize)
    real_grid = np.real(np.fft.fftn(reciprocal_grid))

    # declare gemmi.FloatGrid object
    float_grid = gemmi.FloatGrid(*gridsize)
    float_grid.set_unit_cell(new_mtz.cell)

    if spacegroup is not None:
        float_grid.spacegroup = gemmi.find_spacegroup_by_name(spacegroup)
    else:
        float_grid.spacegroup = new_mtz.spacegroup

    # write real_grid into float_grid via buffer protocol
    temp = np.array(float_grid, copy=False)
    temp[:, :, :] = real_grid[:, :, :]

    # Enforce that mean=0, stdev=1 for voxel intensities
    float_grid.normalize()

    return float_grid


def interpolate_maps(fgoff, fgon):
    """
    Interpolate "on" fg onto the voxel frame of "off" fg

    Parameters
    ----------
    fgoff : gemmi.FloatGrid
        Map with the desired voxel frame
    fgon : gemmi.FloatGrid
        Map to be interpolated onto the fgoff voxel frame

    Returns
    -------
    fgon_interp : gemmi.FloatGrid
        A map with the same contents as fgon, but interpolated onto the voxel frame of fgoff
    """

    gridsize = fgoff.shape

    # rather than declare a new FloatGrid, just clone the old one and fill it with 0s
    # The gridsize, unit_cell, etc. are interited from fg1 as desired
    fgon_interp = fgoff.clone()
    fgon_interp.fill(0)

    # TO-DO: contact the gemmi devs to see if there is a way to vectorize this calculation
    # Meanwhile, we have this silly loop, which takes between 30 seconds and 2 minutes depending on grid size
    for a in tqdm(range(gridsize[0])):
        for b in range(gridsize[1]):
            for c in range(gridsize[2]):
                (
                    fgon_interp.set_value(
                        a, b, c, fgon.interpolate_value(fgoff.get_position(a, b, c)),
                    )
                )

    return fgon_interp


def ilk_from_numpy(ref, mov, **kwargs):
    """
    Perform optical_flow_ilk registration on two numpy arrays

    Parameters
    ----------
    ref : np.array
        "Reference" numpy array that will not get moved
    mov : np.array
        "Moving" numpy array to be registered onto `ref`. Must have same shape as `ref`.
    **kwargs : any
        kwargs will be passed to skimage.registration.optical_flow_ilk.
        Possible kwargs include radius (float), num_warp (int), and gaussian (bool).
        See details at https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.optical_flow_ilk

    Returns
    -------
    mov_reg : np.array
        Result of registering `mov` onto `ref`
    (flow_x, flow_y, flow_z) : tuple
        Tuple representing flow field
    """

    # output of the actual registration are three "flow" vectors
    flow_x, flow_y, flow_z = optical_flow_ilk(ref, mov, **kwargs)

    # boilerplate
    a, b, c = ref.shape
    x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(a), np.arange(b), np.arange(c), indexing="ij"
    )

    # Flow vectors define the warp needed to get mov_reg from mov
    mov_reg = warp(
        mov,
        np.array([x_coords + flow_x, y_coords + flow_y, z_coords + flow_z]),
        mode="edge",
    )

    return mov_reg, (flow_x, flow_y, flow_z)


def register_maps(
    mtzoff,
    mtzon,
    pdboff,
    mapnameoff,
    mapnameon,
    diffmapname,
    Foff="FP",
    Phioff="PHWT",
    Fon="F-obs",
    Phion="PH2FOFCWT",
    path="./",
    radius=14,
    num_warp=5,
    gaussian=False,
    spacing=0.25,
    spacegroup="P1",
    on_as_stationary=False,
    python_returns=False,
):
    """
    Perform optical-flow-based map registration. Take in two mtz files, and return three map files corresponding to the first mtz,
    a registered version of the second mtz, and a difference map.

    Parameters
    ----------
    mtzoff : rs.DataSet
        input mtz representing the off/apo/ground/dark state
    mtzon : rs.DataSet
        input mtz representing the on/bound/excited/bright state
    pdboff : gemmi.Structure
        input pdb used to find the bounding box on which registration is performed
    mapnameoff : str
        Name of output map containing off/apo/ground/dark data
    mapnameon : str
        Name of output map containing on/bound/excited/bright data
    diffmapname : str
        Name of output difference map containing the registered difference on - off
    path : str, optional
        file location of mtzoff and mtzon and to write out maps, by default './'
    Foff : str, optional
        Name of column in mtzoff containing desired structure factor amplitudes, by default "FP" (assuming an mtz downloaded from the pdb)
    Phioff : str, optional
        Name of column in mtzoff containing desired phases, by default "PHWT" (assuming an mtz downloaded from the pdb)
    Fon : str, optional
        Name of column in mtzon containing desired structure factor amplitudes, by default "F-obs" (assuming phenix.refine output)
    Phion : str, optional
        Name of column in mtzon containing desired phases, by default "PH2FOFCWT" (assuming phenix.refine output)
    radius : int, optional
        Optional argument to pass to optical_flow_ilk determining the radius (in pixels) considered around each pixel, by default 14
    num_warp : int, optional
        Optional argument to pass to optical_flow_ilk determining the number of time to iterate registration, by default 5
    gaussian : bool, optional
        Optional argument to pass to optical_flow_ilk to use a gaussian kernel (True) or uniform kernel (False), by default False
    spacing : float, optional
        Approximate voxel size in Angstroms of the output maps, by default 0.25
    on_as_stationary: bool, optional
        If True, register off data onto on data. Useful if ligands are modeled in on data
        If False (default) register on data onto off data.
    python_returns : bool, optional
        If True, return a 4-tuple of the off FloatGrid, on FloatGrid, registration flow, and bottom corner of the protein model. Do not write out maps.
        If False (default) write out map files and return nothing

    """
    print("Constructing FloatGrids from mtzs...")
    fg_off = make_floatgrid(mtzoff, spacing, F=Foff, Phi=Phioff, spacegroup=spacegroup)
    fg_on = make_floatgrid(mtzon, spacing, F=Fon, Phi=Phion, spacegroup=spacegroup)
    print("Constructed FloatGrids from mtzs")

    print("Interpolating 'on' grid onto 'off' grid frame...")
    fg_on_interpolated = interpolate_maps(fg_off, fg_on)
    print("Interpolated 'on' grid onto 'off' grid frame")

    print("Performing optical flow - this may take up to ~10 minutes")

    # this is where bounding box stuff is computed
    # this code chunk needs operate on fg_on_interpolated and fg_off, and output array_on and array_off

    box = pdboff.calculate_box()
    pad = 10  # e.g. a 2.5A-ish pad

    n = fg_off.get_nearest_point(box.minimum)
    x = fg_off.get_nearest_point(box.maximum)

    true_bottom = (n.u, n.v, n.w)
    true_top = (x.u, x.v, x.w)

    padded_bottom = [b - pad for b in true_bottom]
    padded_top = [t + pad for t in true_top]

    grid_size = fg_off.shape

    shape_for_padded_array = [
        pt - pb if tt > tb else pt - pb + s
        for tb, tt, pb, pt, s in zip(true_bottom, true_top, padded_bottom, padded_top, grid_size)
    ]

    print(f"{n=}\n {x=}\n {true_bottom=}\n {true_top=}\n {padded_bottom=}\n {padded_top=}\n {shape_for_padded_array=}")

    padded_array_off = fg_off.get_subarray(
        start=[*padded_bottom], shape=[*shape_for_padded_array]
    )
    padded_array_on = fg_on_interpolated.get_subarray(
        start=[*padded_bottom], shape=[*shape_for_padded_array]
    )

    # array_off = fg_off.array
    # array_on = fg_on_interpolated.array

    if on_as_stationary:
        padded_array_off, flow = ilk_from_numpy(
            ref=padded_array_on,
            mov=padded_array_off,
            radius=radius,
            num_warp=num_warp,
            gaussian=gaussian,
        )

    else:
        padded_array_on, flow = ilk_from_numpy(
            ref=padded_array_off,
            mov=padded_array_on,
            radius=radius,
            num_warp=num_warp,
            gaussian=gaussian,
        )

    print("Performed optical flow")

    # return array_on, array_off to correct places in fg_on_interpolated, fg_off
    array_off = padded_array_off[pad:-pad, pad:-pad, pad:-pad]
    array_on = padded_array_on[pad:-pad, pad:-pad, pad:-pad]

    print(f"{array_off.shape=}\n {array_on.shape=}\n {padded_array_off.shape=}\n {padded_array_on.shape=}")

    fg_off.set_subarray(array_off, start=[*true_bottom])
    fg_on_interpolated.set_subarray(array_on, start=[*true_bottom])

    fg_diff = fg_off.clone()
    fg_diff.fill(0)
    fg_diff.set_subarray(array_on - array_off, start=[*true_bottom])

    if python_returns:
        return (fg_on_interpolated, fg_off, flow, true_bottom)

    rs.io.write_ccp4_map(
        fg_off.array, f"{path}{mapnameoff}.map", fg_off.unit_cell, fg_off.spacegroup
    )
    rs.io.write_ccp4_map(
        fg_on_interpolated.array,
        f"{path}/{mapnameon}.map",
        fg_off.unit_cell,
        fg_off.spacegroup,
    )
    rs.io.write_ccp4_map(
        fg_diff.array, f"{path}/{diffmapname}.map", fg_off.unit_cell, fg_off.spacegroup,
    )
    print("Wrote map files")
    return


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(description="Parse arguments for map registration")

    parser.add_argument(
        "--mtzoff",
        "-f",
        nargs=3,
        metavar=("mtzfileoff", "Foff", "Phioff"),
        required=True,
        help=(
            "Reference mtz representing off/apo/ground/dark state"
            "Specified as (filename, F, Phi)"
        ),
    )

    parser.add_argument(
        "--mtzon",
        "-n",
        nargs=3,
        metavar=("mtzfileon", "Fon", "Phion"),
        required=True,
        help=(
            "mtz representing the on/bound/excited/bright state"
            "Specified as (filename, F, Phi)"
        ),
    )

    parser.add_argument(
        "--pdboff",
        "-p",
        required=True,
        help=(
            "Reference pdb corresponding to the off/apo/ground/dark state. "
            "Map registration will be performed on the region containing protein "
            "as per calling gemmi's pdb.calculate_box() on this pdb."
        ),
    )

    parser.add_argument(
        "--mapnames",
        "-m",
        nargs=3,
        required=True,
        metavar=("mapnameoff", "mapnameon", "diffmapname"),
        help="Names for off map, on map, and (on - off) difference map",
    )

    parser.add_argument(
        "--on-as-stationary",
        required=False,
        action="store_true",
        default=False,
        help="Include this flag to register 'off' onto 'on' (instead of 'on' onto 'off', the default)",
    )

    parser.add_argument(
        "--path",
        required=False,
        default="./",
        help="Path to mtzs and to which maps should be written. Optional, defaults to './' (current directory)",
    )

    parser.add_argument(
        "--radius",
        required=False,
        type=int,
        default=14,
        help=(
            "Window around a pixel to be used for estimating optical flow, by default 14. "
            "See https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.optical_flow_ilk for details on underlying scikit-image function. "
            "See https://en.wikipedia.org/wiki/Lucas-Kanade_method for details on the Lucas-Kanade registration algorithm"
        ),
    )

    parser.add_argument(
        "--gaussian",
        "-g",
        action="store_true",
        default=False,
        help=(
            "Include this flag to use a gaussian kernel for the Lucas-Kanade algorithm, rather than uniform (default). "
            "See https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.optical_flow_ilk for details on underlying scikit-image function. "
            "See https://en.wikipedia.org/wiki/Lucas-Kanade_method for details on the Lucas-Kanade registration algorithm"
        ),
    )

    parser.add_argument(
        "--num_warp",
        "-w",
        required=False,
        type=int,
        default=5,
        help=(
            "Number of times to iterate Lucas-Kanade image registration. By default, 5. "
            "See https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.optical_flow_ilk for details on underlying scikit-image function. "
            "See https://en.wikipedia.org/wiki/Lucas-Kanade_method for details on the Lucas-Kanade registration algorithm"
        ),
    )

    parser.add_argument(
        "--spacing",
        "-s",
        required=False,
        type=float,
        default=0.25,
        help=(
            "Approximate voxel size in Angstroms. Defaults to 0.25 A. "
            "Value is approximate because there must be an integer number of voxels along each unit cell dimension"
        ),
    )

    parser.add_argument(
        "--spacegroup",
        required=False,
        default="P1",
        help=(
            "Spacegroup into which real-space maps will be coerced. By default, P1. "
            "Must be a valid call to the gemmi.UnitCell() constructor."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    # only read from file in the command-line function, not the python function
    mtzoff = rs.read_mtz(args.path + args.mtzoff[0])
    mtzon = rs.read_mtz(args.path + args.mtzon[0])
    pdboff = gemmi.read_structure(args.path + args.pdboff)

    register_maps(
        mtzoff=mtzoff,
        mtzon=mtzon,
        pdboff=pdboff,
        mapnameoff=args.mapnames[0],
        mapnameon=args.mapnames[1],
        diffmapname=args.mapnames[2],
        path=args.path,
        Foff=args.mtzoff[1],
        Phioff=args.mtzoff[2],
        Fon=args.mtzon[1],
        Phion=args.mtzon[2],
        radius=args.radius,
        num_warp=args.num_warp,
        gaussian=args.gaussian,
        spacing=args.spacing,
        spacegroup=args.spacegroup,
        on_as_stationary=args.on_as_stationary,
        python_returns=False,
    )

    return


if __name__ == "__main__":
    main()
