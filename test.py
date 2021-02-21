import argparse
import configparser

parser = argparse.ArgumentParser()
parser.add_argument("--ini_file")
args = parser.parse_args()

cfg = configparser.ConfigParser()
cfg.read(args.ini_file)