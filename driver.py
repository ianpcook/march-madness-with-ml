import datetime as dt
from data_prepper import data as dp
import glob
import numpy as np
import os
import pandas as pd

os.chdir('C:\\Users\\MeadeHouse\\PycharmProjects\\mm2018\\march-madness-with-ml')

def build_dataset():
    tourney_year = dt.datetime.now().year
    trainingX, trainingY, team_stats = dp.get_data()

def get_models():
    # get the name of the models that are going to be run
    model_list = [os.path.basename(x).split('.')[0] for x in glob.glob('./src/*py')]