def getVersionNumber():
    import pkg_resources

    version = pkg_resources.require("mapreg")[0].version
    return version


__version__ = getVersionNumber()

from .register_maps import register_maps, ilk_from_numpy
from .prep_for_registration import prep_for_registration