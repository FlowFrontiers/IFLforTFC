# Incremental Federated Learning for Traffic Flow Classification in Heterogeneous Data Scenarios

This is the supporting page for the manuscript titled, _"Incremental Federated Learning for Traffic Flow Classification in Heterogeneous Data Scenarios."_

The paper presents a comparative study of Federated Learning models, developed under both IID and non-IID conditions, and a Centralized Learning model. The objective is to perform multi-class traffic flow classification for network applications.

In line with principles of reproducibility and transparency, and to support open science, the scripts utilized in the experiments are available to the public. Additionally, the dataset generated for this research is also accessible. These resources aim to aid a more comprehensive understanding of the methodologies employed and foster additional research in this field.

Researchers are encouraged to use this dataset for their investigations and to replicate or extend the findings of this study. The availability of this dataset is expected to support continued research in network flow analysis, thus adding to the body of knowledge in this area.

---

> **📥 Dataset Access**  
> **Need the dataset? Download it directly from:**  
> **http://pekar.s.cnl.sk/IFLforTFC/dataset.parquet**  
> *(Alternative link if repository access is unavailable)*

---

## Repository Structure

In our repository, the files and scripts are organized as follows:
 - [1-prepare-datasets.ipynb](1-prepare-datasets.ipynb): houses the scripts for application label encoding, feature selection, and dataset preparation for both centralized and federated learning models. Additionally, this file contains the code used to segment data into IID and non-IID chunks for federated learning. It also features scripts for visualizing data distribution across various application types, both for the full dataset and individual chunks.
 - [2-evaluate-cl.ipynb](2-evaluate-cl.ipynb): incorporates the code used to execute centralized learning and visualize the obtained results, which are stored as a JSON file at [results/cl_results.json](results/cl_results.json). In this file, you can see how the centralized model is trained and tested, and the process of result visualization.
 - [3-evaluate-fl-iid.ipynb](3-evaluate-fl-iid.ipynb): contains the functions employed for a visual comparison of FL results under IID conditions, stored at [results/iid_fl_*.json](results/), against the CL results found at [results/cl_results.json](results/cl_results.json). It offers a comparative analysis between FL (under IID condition) and CL results.
 - [4-evaluate-fl-iid-shuffled.ipynb](4-evaluate-fl-iid-shuffled.ipynb): contains the functions employed for a visual comparison of FL results under IID conditions, stored at [results/iid_fl_shuffled_*.json](results/), against the CL results found at [results/cl_results.json](results/cl_results.json). Compared to the IID scenario, in this case, the dataset was shuffled prior chunking. This notebook offers a comparative analysis between FL (under a different IID condition) and CL results.
 - [5-evaluate-fl-non-iid-A.ipynb](4-evaluate-fl-non-iid-A.ipynb): holds the functions used for visual comparison of FL results achieved under non-IID _scenario A_ conditions, which are stored at [results/non_iid_A_fl_*.json](results/), against the CL results found at [results/cl_results.json](results/cl_results.json). This notebook will helped to compare and understand the performance differences between FL (under non-IID-A condition) and CL models.
 - [6-evaluate-fl-non-iid-B.ipynb](5-evaluate-fl-non-iid-B.ipynb): holds the functions used for visual comparison of FL results achieved under non-IID _scenario B_ conditions, which are stored at [results/non_iid_B_fl_*.json](results/), against the CL results found at [results/cl_results.json](results/cl_results.json). This notebook will helped to compare and understand the performance differences between FL (under non-IID-B condition) and CL models.
 - [7-evaluate-fl-non-iid-C.ipynb](6-evaluate-fl-non-iid-C.ipynb): holds the functions used for visual comparison of FL results achieved under non-IID _scenario C_ conditions, which are stored at [results/non_iid_C_fl_*.json](results/), against the CL results found at [results/cl_results.json](results/cl_results.json). This notebook will helped to compare and understand the performance differences between FL (under non-IID-C condition) and CL models.
 - [fl-client.py](fl-client.py): Is a python program that facilitates FL tasks on the client side.
 - [fl-server.py](fl-server.py): Is a python program that facilitates FL tasks on the serer side.

The Jupyter notebooks are designed to be self-explanatory, guiding the reader through each step of the process. However, if any issues are encountered or something is unclear, it is encouraged to raise an issue on GitHub. Contributions to the improvement of this repository are greatly appreciated.

The repository also contains some helper functions primarily used for validity check and visualisaiton.

## FL Dataset Conditions Studied 

In our study, we examine the following conditions for the input dataset chunks:

|                                                   | IID A | non-IID A | non-IID B | non-IID C |
|---------------------------------------------------|-------|-----------|-----------|-----------|
| Each chunk has identical sample size              | 1     | 0         | 0         | 0         |
| Each chunk has identical application distribution | 1     | 1         | 0         | 0         |
| Each chunk has identical application count        | 1     | 1         | 1         | 0         |


## Federated Learning

The Federated Learning (FL) models used in our study were built using the FeNOMan tool. The tool encompasses both server and client-side components written in Python.

### Setting Up the FL Server

To execute the server-side code, use the following command:

```bash
python3 fl-server.py --scenario non_iid_A
```

The `--scenario` argument specifies the evaluation scenario and can be set to one of the following values: 
- `iid`
- `non_iid_A`
- `non_iid_B`
- `non_iid_C`

### Setting Up the FL Client

To launch the client-side code, execute:

```bash
python3 fl-client.py --client_id c1 --scenario non_iid_A 
```

The `--client_id` argument indicates the unique client name, which is later reflected in the JSON output filenames used for data visualization. The choice for the `--scenario` argument remains consistent with the server-side options, i.e., `iid`, `iid_shuffled`, `non_iid_A`, `non_iid_B`, or `non_iid_C`.

To generate the input files for this study, refer to the [1-prepare-datasets.ipynb](1-prepare-datasets.ipynb) script.

## Dataset

The dataset necessary to execute these notebooks are available at [datasets/dataset.parquet](datasets/dataset.parquet).

> **📥 Alternative Download Link**  
> **If you experience issues accessing the dataset from the repository, you can download it directly from:**  
> **http://pekar.s.cnl.sk/IFLforTFC/dataset.parquet**

## Documentation

For a deeper understanding of the project, reference can be made to the related research paper. There, additional insights about the methodologies employed can be found.

We hope this repository serves as a valuable resource in related research.
