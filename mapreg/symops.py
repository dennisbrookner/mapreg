
# def apply_symop_to_map(mtz, op, parse_string='True'):
#     """
#     Quick quasi-commandline-utility to apply a symmetry operation to an mtz

#     Parameters
#     ----------
#     mtzin : str
#         Path and filename of input mtz
#     mtzout : str
#         Path and filename of output mtz to be saved
#     op : str
#         String representing a symmetry operation
#         Must be a valid argument to the gemmi.Op() constructor, unless parse_string is True
#     parse_string : bool, optional
#         If True, parse string via symop_parser
#     python_returns : bool, optional
#         If True return mtzout as a python object, by default False

#     """
#     mtz = rs.read_mtz(mtzin)

#     if parse_string=='True':
#         op = symop_parser(op)

#     mtz_after_symop = mtz.apply_symop(gemmi.Op(op))

#     mtz_after_symop.write_mtz(mtzout)

#     if python_returns:
#         return mtz_after_symop
#     else:
#         return


def symop_parser(string):
    """
    Create a valid symmetry operation from a phenix find alternate symmetry origins log file

    Parameters
    ----------
    string : str
        should be output of `> grep symm.op logfile`, for example:
        6. symm.op: (y,x,-z), 2. origin: (0, 0, 1/2), UC shift: (0, 1, 0), 1. SSM Q: 0.879, A

    Returns
    -------
    op : str
        string that is a valid input for the gemmi.Op constructor
    """

    symm = string.split(" ")[2][1:-2].split(",")
    origin = " ".join(string.split(" ")[5:8])[1:-2].split(", ")
    op = ",".join(list([x + "+" + str(y) for x, y in zip(symm, origin)]))
    return op
