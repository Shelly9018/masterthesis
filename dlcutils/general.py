import platform


class WinLinuxPathReplacer:
    """
    Converts strings to match linux and windows path descriptions. Assumes that paths are
    generally identical with the difference that base folders/drive names may be different.
    Replaces the linux path with the given windows equivalent if the submitted string starts
    with one of the strings indicated by the path strings.

    Parameters:
    conversion_dict(dict, optional): contains key(linux)/value(windows) path pairs

    Returns:
    p (str): the path with replaced beginning if beginning was in the dict keys
    """

    def __init__(self, conversion_dict=None):

        self.conversion_dict = conversion_dict

        if self.conversion_dict is None:
            self.conversion_dict = {
                "/images": "//filesrv/images",
                "/work/scratch": "//filesrv/scratch"
            }

    def __call__(self, p):

        if platform.system() == "Windows":

            for k in self.conversion_dict.keys():

                if p.startswith(k):
                    p = p.replace(k, self.conversion_dict[k], 1)

        return p

