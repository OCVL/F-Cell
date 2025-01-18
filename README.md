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
The configuration file uses a json file format. At the moment, its creation is manual, though we will be developing a GUI tool for easy creation/updating of parameters in the coming months.

The base format of the configuration json has the following structure:

```json
{
  "version": "0.2",
  "description": "The pipeline and analysis JSON for the OCVL's MEAOSLO.",
  "recursive_search": true
  "raw": { }
  "processed": { }
  "pipelined": { }
}
```

This corresponds to the following key/value pairs: 

- `version: "string"`:  The version of the configuration file used. The current newest version is 0.2
- `description: "string"`: The description of the configuration. Useful if multiple configurations are used for your particular analysis, if you have multiple devices, or if you want to test multiple pipeline/analysis combinations.
- `recursive_search: [true/false]`: Whether or not to analyze the folder structure recursively from the user-selected folder.
- `raw: {}`: Parameters relating to handling of **raw** data
- `processed: {}`: Parameters relating to handling of data that has been registered, or **processed**.
- `pipelined: {}`: Parameters relating to handling of data that has gone through F-Cell's pre-processing pipeline.



### Pre-processing pipeline:

### Analysis:
