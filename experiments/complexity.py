# %%
from ldimbenchmark.methods import LILA, MNF, DUALMethod
from BattLeDIM import LDIMBenchmark

methods = [MNF(), LILA(), DUALMethod()]


benchmark = LDIMBenchmark({}, [], results_dir="./benchmark-results")
benchmark.add_local_methods(methods)

results = benchmark.run_complexity_analysis(
    methods=methods,
    style="time",
)


results


# benchmark.run_complexity_analysis(
#     methods=methods,
#     style="junctions",
# )

# %%
