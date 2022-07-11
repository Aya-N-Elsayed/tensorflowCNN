
import pandas as pd
import numpy as np
from classifiers import *
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds



if __name__ == "__main__":
    
    classi = classifiers()
    classi.run_classifier()