from datasetsforecast.m4 import M4
import pandas as pd

def load_m4_financial(data_dir='./data'):
    groups = ['Daily', 'Monthly', 'Yearly']
    for group in groups:
        M4.download(data_dir, group=group)
