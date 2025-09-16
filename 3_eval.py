
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import config
from classes import FaissIndex, MetricsLogger, Reranker, MyDataSet
import time
import logging
import json
from datetime import datetime

from utils import init_logging











def main():
    ds = MyDataSet(confs=config)
