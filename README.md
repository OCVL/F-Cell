# F-Cell
F(Cell) - Function of the Cell

### Introduction:
This repository is where the OCVL does most of its optoretinography software development. This software will always be open source, and available to any researcher using it for non-profit work. For the moment, the software is designed around intensity-based optoretinography, or iORG processing. That means that it is capable of processing/analyzing data from any en-face device that produces videos, such as scanning laser ophthalmoscopes or line-scan ophthalmoscopes, and their adaptive optics variants (e.g. AO-SLO/AO-LSO/AO-FiO).

### Attribution:
In the future, if you use this software you will be able to cite a paper referring to this repository. In the meantime, please cite Gaffney et. al., "Intensity-based optoretinography reveals sub-clinical deficits in cone function in retinitis pigmentosa", 2024 - the algorithms and software used in that paper mirror those seen here.

## Using the software:
Working with optoretinograms often requires supra-normal processing of AOSLO/AOLSO/AOOCT datasets, so we have broken optoretinogram generation into a "pipeline" stage and an "analysis" stage. 

How these stages work are governed by json-based configuration files that allow you to run the code on your data's particular filename, video, and metadata format. It also allows you to specify the parameters used during the processing and analysis steps, for your specific scientific problem.

### Configuration files:
The configuration file uses a json file format. At the moment, its creation is manual, though we will be developing a GUI tool for easy creation/updating of parameters in the coming months. **Note: Examples can be found in the config_files directory.**

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

This corresponds to the following key/value pairs, where options for each are in parenthesis, e.g: `your_mom: ("is lovely", "wears combat boots")`: 

- `version: "string"`:  The version of the configuration file used. The current newest version is 0.2
- `description: "string"`: The description of the configuration. Useful if multiple configurations are used for your particular analysis, if you have multiple devices, or if you want to test multiple pipeline/analysis combinations.
- `raw: {}`: Parameters relating to handling of raw data. **Currently unused.**
- `processed: {}`: Parameters relating to handling of data that has been registered, or **processed**.
- `pipelined: {}`: Parameters relating to handling of data that has gone through F-Cell's pre-processing pipeline.

#### `processed` keys/values:
The processed parameters are as follows.

- `video_format: "format string"`: The filename format of the **video** (e.g. avi, mp4, etc) associated with a single acquisition. Uses tag formatting to extract file-specific metadata.
- `mask_format: "format string"`: The filename format of the **video of masks** associated with the video. Consists of a video of binary masks that describes the valid region in the video. Uses tag formatting to extract file-specific metadata.
- `image_format: "format string"`: The filename format of the averaged image associated with a single acquisition. Uses tag formatting to extract file-specific metadata.
- `recursive_search: (true/false)`: Whether or not to analyze the folder structure recursively from the user-selected folder.
- `metadata: {}`: Parmaeters pertaining to a datasets' metadata.
  - `type: ("text_file", "mat_file", "database")`: The source of non-filename based metadata.
  - `metadata_format: "format string"`: The filename format, or database path of the metadata associated with a single acquisition. Uses tag formatting to extract file-specific metadata.
  - `fields_to_load: {}`: The fields to load from the metadata source. If not specified, will load all metadata from the file.
    Currently internally handled fields are:
    - `framestamps: "string"`: The original frame indices, or "framestamps" of the processed video data. Included to track which frames were dropped from the original video.
    - `stimulus_sequence: "string"`: The stimulus status (on/off), per frame, of the video. Length must match framestamps, and be a column of true/false values.


### Pre-processing pipeline:

### Analysis:
