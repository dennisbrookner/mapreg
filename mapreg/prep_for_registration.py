"""
Prep inputs for registration
"""

import argparse
import shutil
import subprocess
import glob

import reciprocalspaceship as rs
import gemmi


def rigid_body_refine(mtzon, pdboff, path, ligands=None, eff=None, verbose=False, selection=None):

    if eff is None:
        eff_contents = """
refinement {
  crystal_symmetry {
    unit_cell = cell_parameters
    space_group = sg
  }
  input {
    pdb {
      file_name = pdb_input
    }
    xray_data {
      file_name = "mtz_input"
      labels = FPH1,SIGFPH1
      r_free_flags {
        generate=True
      }
      force_anomalous_flag_to_be_equal_to = False
    }
    monomers {
      ligands
    }
  }
  output {
    prefix = '''nickname'''
    serial = 1
    serial_format = "%d"
    job_title = '''nickname'''
    write_def_file = False    
    write_eff_file = False
    write_geo_file = False
  }
  electron_density_maps {
    map_coefficients {
      map_type = "2mFo-DFc"
      mtz_label_amplitudes = "2FOFCWT"
      mtz_label_phases = "PH2FOFCWT"
      fill_missing_f_obs = True
    }
  }
  refine {
    strategy = *rigid_body 
    sites {
      rigid_body = all
    }  
  }
  main {
    number_of_macro_cycles = 1
    nproc = 8
  }
}
    """
    else:
        with open(eff, 'r') as file:
            eff_contents = file.read()

    nickname = f"{mtzon.removesuffix('.mtz')}_rbr_to_{pdboff.removesuffix('.pdb')}"
    
    # check existing files because phenix doesn't like to overwrite things
    similar_files = glob.glob(f'{nickname}_[0-9]_1.*')
    if len(similar_files) == 0:
        nickname += '_0'
    else:
        n = max([int(s.split('_')[-2]) for s in similar_files])
        nickname += f'_{n+1}'


    # read in mtz to access cell parameters and spacegroup
    mtz = rs.read_mtz(path + mtzon)
    cell_string = f"{mtz.cell.a} {mtz.cell.b} {mtz.cell.c} {mtz.cell.alpha} {mtz.cell.beta} {mtz.cell.gamma}"
    sg = mtz.spacegroup.short_name()

    # edit refinement template
    eff = f"params_{nickname}.eff"

    params = {
        "sg": sg,
        "cell_parameters": cell_string,
        "pdb_input": path + pdboff,
        "mtz_input": path + mtzon,
        "nickname": nickname,
    }
    
    if selection is not None:
        params['all'] = selection # overwrite atom selection
    
    for key, value in params.items():
        eff_contents = eff_contents.replace(key, value)

    # either add ligands to .eff file or delete "ligands" placeholder
    if ligands is not None:
        ligand_string = "\n".join([f"file_name = '{l}'" for l in ligands])
        eff_contents = eff_contents.replace("ligands", ligand_string)
    else:
        eff_contents = eff_contents.replace("ligands", "")

    # write out customized .eff file for use by phenix
    with open(eff, "w") as file:
        file.write(eff_contents)

    # confirm that phenix is active in the command-line environment
    if shutil.which("phenix.refine") is None:
        raise EnvironmentError(
            "Cannot find executable, phenix.refine. Please set up your phenix environment."
        )

    # run refinement!
    # print refinement output to terminal if user supplied the --verbose flag
    subprocess.run(
        f"phenix.refine {eff}",
        shell=True,
        capture_output=(not verbose),
    )

    print(f"Ran phenix.refine and produced {nickname}_1.mtz")
    print(f"Use this file as --mtzon for mapreg.register ")

    return  


def handle_special_positions(pdboff, path):
    """
    Check if any waters happen to sit on special positions, and if so, remove them.
    If any non-water atoms sit on special positions, throw a (hopefully helpful) error.

    Parameters
    ----------
    pdboff : str
        name of input pdb
    path : str
        Relative in/out path for files
    """
    pdb = gemmi.read_structure(path + pdboff)
    
    # gemmi pdb heirarchy is nonsense, but this is what we've got
    for model in pdb:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    
                    # check if atom is within 0.5 A of a special position
                    # returns a number > 0 if so
                    if pdb.cell.is_special_position(atom.pos, max_dist=0.5) > 0: 
                        if residue.is_water():
                            print('Input model contains water at special position. Removing water so as to not break rigid-body refinement.')
                            print('If it is important that you keep this water and just omit it from your refinement selection, use the --selection flag')
                            residue.remove_atom(name=atom.name, altloc=atom.altloc, el=atom.element)
                            
                        else:
                            raise ValueError("""
Input model contains a non-water atom on a special position. 
Please use the --selection flag to supply an input suitable for rigid-body refinement by phenix. 
""")
    
    
    return pdboff


def prep_for_registration(
    pdboff,
    mtzoff,
    mtzon,
    ligands=None,
    Foff="FP",
    SigFoff="PHWT",
    Fon="FP",
    SigFon="PHWT",
    path="./",
    verbose=False,
    selection=None,
    # symop=None,
    eff=None
):

    # if symop is not None:
    #     mon = rs.read_mtz(path + mtzon)
    #     mon.apply_symop(symop, inplace=True)
    #     mtzon = mtzon.removesuffix(".mtz") + "_reindexed" + ".mtz"

    #     mon.write_mtz(path + mtzon)

    mtzon_scaled = mtzon.removesuffix(".mtz") + "_scaled" + ".mtz"

    subprocess.run(
        f"rs.scaleit -r {mtzoff} {Foff} {SigFoff} -i {mtzon} {Fon} {SigFon} -o {mtzon_scaled}",
        shell=True,
        capture_output=(not verbose),
    )
    print(f"Ran scaleit and produced {mtzon_scaled}")
    
    pdboff = handle_special_positions()

    mtzon = mtzon_scaled
    
    print(f"Running phenix.refine...")

    rigid_body_refine(
        mtzon=mtzon,
        pdboff=pdboff,
        path=path,
        ligands=ligands,
        eff=eff,
        verbose=verbose,
        selection=selection,
    )

    return


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare inputs for map registration. "
            "Note that both ccp4 and phenix must be active in the environment. "
        )
    )

    parser.add_argument(
        "--pdboff",
        "-p",
        required=True,
        help=(
            "Reference pdb corresponding to the off/apo/ground/dark state. "
            "Used to rigid-body refine onto `mtzon` and obtain 'on' phases."
        ),
    )

    parser.add_argument(
        "--ligands",
        "-l",
        required=False,
        default=None,
        nargs="*",
        help=("Any .cif restraint files needed for refinement"),
    )

    parser.add_argument(
        "--mtzoff",
        "-f",
        nargs=3,
        metavar=("mtzfileoff", "Foff", "SigFoff"),
        required=True,
        help=(
            "Reference mtz containing off/apo/ground/dark state data. "
            "Specified as (filename, F, SigF)"
        ),
    )

    parser.add_argument(
        "--mtzon",
        "-n",
        nargs=3,
        metavar=("mtzfileon", "Fon", "SigFon"),
        required=True,
        help=(
            "mtz containing on/bound/excited/bright state data. " 
            "Specified as (filename, F, SigF)"
            ),
    )

    parser.add_argument(
        "--path",
        required=False,
        default="./",
        help="Path to mtzs and to which maps should be written. Optional, defaults to './' (current directory)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        required=False,
        action="store_true",
        default="False",
        help="Include this flag to print out scaleit and phenix.refine outputs to the terminal",
    )

    # parser.add_argument(
    #     "--symop",
    #     required=False,
    #     default=None,
    #     help=("Symmetry operation for reindexing mtz2 to match mtz1"),
    # )
    
    parser.add_argument(
        "--selection",
        required=False,
        default=None,
        help=("Argument to phenix.refine's refinement.refine.sites.rigid_body. "
              "Use this flag if atoms on special positions need to be omitted from the atom selection. "),
    )

    parser.add_argument(
        "--eff",
        required=False,
        default=None,
        help=("Custom .eff file for running phenix.refine. "),
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    prep_for_registration(
        pdboff=args.pdboff,
        ligands=args.ligands,
        mtzoff=args.mtzoff[0],
        mtzon=args.mtzon[0],
        Foff=args.mtzoff[1],
        SigFoff=args.mtzoff[2],
        Fon=args.mtzon[1],
        SigFon=args.mtzon[2],
        path=args.path,
        verbose=args.verbose,
        # symop=args.symop,
        selection=args.selection,
        eff=args.eff
    )

    return


if __name__ == "__main__":
    main()
