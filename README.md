# F-Cell
F(Cell) - Function of the Cell

### Introduction:
This repository is where the OCVL does most of its optoretinography software development. This software will always be open source, and available to any researcher using it for non-profit work. For the moment, the software is designed around intensity-based optoretinography, or iORG processing. That means that it is capable of processing/analyzing data from any en-face device that produces videos, such as scanning laser ophthalmoscopes or line-scan ophthalmoscopes, and their adaptive optics variants (e.g. AO-SLO/AO-LSO/AO-FiO).

### Attribution:
In the future, if you use this software you will be able to cite a paper referring to this repository. In the meantime, please cite Gaffney et. al., "Intensity-based optoretinography reveals sub-clinical deficits in cone function in retinitis pigmentosa", 2024 - the algorithms and software used in that paper mirror those seen here.

## Using the software:
Working with optoretinograms often requires supra-normal processing of AOSLO/AOLSO/AOOCT datasets, so we have broken optoretinogram generation into a "pipeline" stage and an "analysis" stage. 

How these stages work are governed by json-based configuration files that allow you to run the code on your data's particular filename, video, and metadata format. It also allows you to specify the parameters used during the processing and analysis steps, for your specific scientific problem.

### Configuration file format:

### Pre-processing pipeline:

### Analysis:
