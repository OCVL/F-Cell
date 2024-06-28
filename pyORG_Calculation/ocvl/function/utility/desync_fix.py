from pathlib import Path

import pandas as pd

from ocvl.function.utility.resources import load_video, save_video



if __name__ == "__main__":

    # Parse out the locations and filenames, store them in a hash table.
    searchpath = Path("//134.48.93.176/Raw Study Data/00-36828/FFB_IndivInvest_2020_PRO38673/MEAOSLO1/20240620")

    for path in searchpath.rglob("*.avi"):
        print("Processing: "+path.name)
        if "Confocal" in path.name:

            res = load_video(path.as_posix())
            video_data = res.data[:, :, :-1]
            save_video(path.as_posix(), video_data, res.metadict["framerate"])

            metadata = pd.read_csv( (path.as_posix()[:-3] + "csv") , delimiter=',', encoding="utf-8-sig")
            metadata = metadata[0:-1]
            metadata.to_csv((path.as_posix()[:-3] + "csv"), index=False)


        else:
            res = load_video(path.as_posix())
            video_data = res.data[:, :, 1:]
            save_video(path.as_posix(), video_data, res.metadict["framerate"])

            metadata = pd.read_csv((path.as_posix()[:-3] + "csv"), delimiter=',', encoding="utf-8-sig")
            metadata = metadata[0:-1]
            metadata.to_csv((path.as_posix()[:-3] + "csv"), index=False)
