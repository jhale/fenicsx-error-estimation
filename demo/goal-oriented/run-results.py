import os
import shutil as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from mpltools import annotation

from demo_goal_oriented import adaptive_refinement


estimators = [("bw", 1, 2),
              ("bw", 2, 4),
              ("res", None, None),
              ("zz", None, None)]

for estimator, param1, param2 in estimators:
    if estimator == "bw":
        estimator_path = "bw_" + str(param1) + "_" + str(param2)
    else:
        estimator_path = estimator

    try:
        if os.path.exists(path=os.path.join("output", estimator_path)):
            st.rmtree(os.path.join("output", estimator_path))
            os.makedirs(os.path.join("output", estimator_path))
        else:
            os.makedirs(os.path.join("output", estimator_path))
    except Exception as e:
        print("Failed to empty or create output dir. Reason:", e)

    adaptive_refinement(estimator, parameters=(param1, param2))
