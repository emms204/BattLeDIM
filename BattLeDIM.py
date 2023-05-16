import sys
sys.path.append('LDIMBenchmark')

from ldimbenchmark.datasets.library import DatasetLibrary, DATASETS
from ldimbenchmark.classes import BenchmarkData, BenchmarkLeakageResult
from ldimbenchmark.benchmark import LDIMBenchmark
# from ldimbenchmark.classes import LDIMMethodBase
from ldimbenchmark.methods.lila import LILA
from typing import List
import logging
logging.basicConfig(level=logging.DEBUG)

datasets = DatasetLibrary("datasets").download(DATASETS.BATTLEDIM)

local_methods = [LILA(modl='LinearRegression')]

hyperparameters = {"default_flow_sensor":"PUMP_1"}

benchmark = LDIMBenchmark(
    hyperparameters, datasets, results_dir="./benchmark-results"
)
benchmark.add_local_methods(local_methods)

benchmark.run_benchmark(evaluation_mode="evaluation")

benchmark.evaluate(write_results="csv")