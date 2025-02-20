# F(Cell) - Function of the Cell

### Introduction:
This repository is where the OCVL does most of its optoretinography software development. This software will always be open source, and available to any researcher using it for non-profit work. For the moment, the software is designed around intensity-based optoretinography, or iORG processing. That means that it is capable of processing/analyzing data from any en-face device that produces videos (not volumes), such as scanning laser ophthalmoscopes or line-scan ophthalmoscopes, and their adaptive optics variants (e.g. AO-SLO/AO-LSO/AO-FiO).

### Attribution:
In the future, if you use this software you will be able to cite a paper referring to this repository. In the meantime, please cite Gaffney et. al., "Intensity-based optoretinography reveals sub-clinical deficits in cone function in retinitis pigmentosa", 2024 - the algorithms and software used in that paper encapsulate those seen here, and this software was used for that paper.

## Using the software:
Working with optoretinograms often requires supra-normal processing of AOSLO/AOLSO/AOOCT datasets, so we have broken optoretinogram generation into a "pipeline" stage and an "analysis" stage. 

For quick start information, [please see our wiki](https://github.com/OCVL/F-Cell/wiki/Quick-Start).

How these stages work are governed by json-based configuration files that allow you to run the code on your data's particular filename, video, and metadata format. It also allows you to specify the parameters used during the processing and analysis steps, for your specific scientific problem.

### Configuration files:
The configuration file uses a json file format. At the moment, its creation is manual, though we will be developing a GUI tool for easy creation/updating of parameters in the coming months. **Note: Examples can be found in the config_files directory.**

The [wiki](https://github.com/OCVL/F-Cell/wiki) containing instructions on how to use the pipeline as well as how to write a configuration file is currently under construction, but keep an eye for more instructions in March.
