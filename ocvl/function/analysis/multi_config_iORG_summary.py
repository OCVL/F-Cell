import gc
import os
import sys
import tracemalloc
from multiprocessing import Process
from pathlib import Path
from tkinter import filedialog
import pandas as pd
from colorama import Fore
import multiprocessing as mp
from ocvl.function.iORG_summary_and_analysis import iORG_summary_and_analysis

if __name__ == "__main__":

    pName = None
    json_fName = Path()
    dat_form = dict()
    allData = pd.DataFrame()

    pName = filedialog.askdirectory(title="Select the folder containing all videos of interest.", initialdir=pName)
    if not pName:
        sys.exit(1)

    # We should be 3 levels up from here. Kinda jank, will need to change eventually
    conf_path = Path(os.path.dirname(__file__)).parent.parent.joinpath("config_files")

    json_path = filedialog.askdirectory(title="Select the configuration json file.", initialdir=conf_path)
    if not json_path:
        sys.exit(2)



    for the_path in Path(json_path).glob("*.json"):

        print(Fore.RED + "\n************ "+str(the_path.name)+" ************\n")
        p = Process(target=iORG_summary_and_analysis, args=(pName, the_path))
        p.start()
        p.join()

        gc.collect()
