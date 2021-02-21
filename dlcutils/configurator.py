from typing import Union
# currently not needed, maybe I will remove it later

class configurator:
    def __init__(self, config_file=None):

        self.args = {}

        if config_file is not None:
            self.load_config_file(config_file)

    def parse_string(self, s: str) -> Union[int, float, str]:
        """
        converts a string into an int, if this fails tries a float
        and if this fails too it returns the original string.
        useful to parse parameters from a config file into a
        suitable numeric data type

        :param s: str, input string
        :return: Union[int, float, str] - the string converted to the best suitable target data type
        """
        try:
            ret = int(s)
        except ValueError:
            try:
                ret = float(s)
            except ValueError:
                ret = s
        return ret


