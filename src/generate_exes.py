import os
import shutil
import tomllib

import PyInstaller.__main__
from PyInstaller.utils.win32 import versioninfo
from PyInstaller.utils.win32.versioninfo import *

if __name__ == "__main__":

    try:
        with open("../pyproject.toml", "rb") as f:
            toml_dict = tomllib.load(f)
            version = toml_dict["project"].get("version", "0.0.0.0")

            v_tuple = tuple(map(int, (version.split('.') + ['0'] * 4)[:4]))


            fileinfo = FixedFileInfo(filevers=v_tuple, prodvers=v_tuple)

            name = StringStruct("CompanyName", "OCVL")
            vers = StringStruct("FileVersion", version)
            pipeint_desc = StringStruct("FileDescription", "The pre-analysis pipeline program for F-Cell.")
            pipeint_name = StringStruct("ProductName", "Pre-analysis pipeline")
            copyright = StringStruct("LegalCopyright", "Copyright (c) 2026. Robert F Cooper")
            analysisint_name = StringStruct("InternalName", "F-Cell analysis")
            analysisint_desc = StringStruct("FileDescription", "F-Cell: Software for reproducible analysis of optoretinograms")
            guiint_name = StringStruct("InternalName", "Configuration gui")
            guiint_desc = StringStruct("FileDescription",
                                            "A GUI for rapidly creating F-Cell-compatible configuration files.")

            strtab = StringTable("040904B0", [name, pipeint_desc, vers, pipeint_name, copyright])
            strfileinfo = StringFileInfo([strtab])

            varstruct = VarStruct("Translation", [1033, 1200])
            vfi = VarFileInfo([varstruct])

            # Make our pipeline exe
            vsvi = VSVersionInfo(fileinfo, [strfileinfo, vfi])

            with open("./ocvl/function/preprocessing/pipe_vinfo.txt", "w") as vf:
                vf.write(str(vsvi))

            PyInstaller.__main__.run([
                 'ocvl/function/preprocessing/pipeline.spec',
                '--noconfirm'
            ])

            strtab = StringTable("040904B0", [name, analysisint_desc, vers, analysisint_name, copyright])
            strfileinfo = StringFileInfo([strtab])
            vsvi = VSVersionInfo(fileinfo, [strfileinfo, vfi])

            # Make our analysis exe
            with open("./ocvl/function/analysis_vinfo.txt", "w") as vf:
                vf.write(str(vsvi))

            shutil.copyfile("../.venv/Lib/site-packages/ssqueezepy/configs.ini", "./ocvl/function/configs.ini")

            PyInstaller.__main__.run([
                 'ocvl/function/iORG_summary_and_analysis.spec',
                '--noconfirm'
            ])

            strtab = StringTable("040904B0", [name, guiint_desc, vers, guiint_name, copyright])
            strfileinfo = StringFileInfo([strtab])
            vsvi = VSVersionInfo(fileinfo, [strfileinfo, vfi])

            # Make our GUI exe
            with open("./ocvl/function/gui/gui_vinfo.txt", "w") as vf:
                vf.write(str(vsvi))

            PyInstaller.__main__.run([
                 'ocvl/function/gui/json_generator.spec',
                '--noconfirm'
            ])

            os.makedirs("./dist/f-cell", exist_ok=True)
            shutil.move("./dist/pipeline/pipeline.exe", "./dist/f-cell/pre_analysis_pipeline.exe")
            shutil.move("./dist/iORG_summary_and_analysis/iORG_summary_and_analysis.exe", "./dist/f-cell/f-cell_analysis.exe")
            shutil.move("./dist/json_generator/json_generator.exe", "./dist/f-cell/f-cell_config_generator.exe")
            shutil.copytree("./dist/pipeline/_internal", "./dist/f-cell/_internal", dirs_exist_ok=True)
            shutil.copytree("./dist/iORG_summary_and_analysis/_internal", "./dist/f-cell/_internal", dirs_exist_ok=True)
            shutil.copytree("./dist/json_generator/_internal", "./dist/f-cell/_internal", dirs_exist_ok=True)


    except FileNotFoundError:
        version = "unknown"