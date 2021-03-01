import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
print('dir', dir_path)
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

from src.model import *
from src.utility import *
import src.dump
import src.train
import src.eval
