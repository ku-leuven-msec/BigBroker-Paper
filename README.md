# Big broker is tracking you! A privacy assessment of large-scale location trace datasets

This repository contains the implementation of the methodology described in the paper "Big broker is tracking you! A privacy assessment of large-scale location trace datasets".

## Abstract

Large-scale location trace datasets are being collected by data brokers and sold for significant financial gains. These datasets are collected continuously through background services in smartphone applications, typically without users’ informed consent. While businesses leverage these data to derive valuable crowd insights, it raises profound privacy concerns for individuals. When such datasets fall into the hands of malicious actors, large-scale blackmail, coercion, and extortion schemes can be set up. This paper demonstrates the risks posed by large-scale location trace datasets, demonstrating how adversaries can exploit them to: (i)  infer sensitive personal locations, such as home and workplace addresses, (ii) reveal the identity behind the traces, and (iii) construct social graphs by linking the location traces of multiple individuals. Our research is unique compared to related work in the sense that it is the first in-depth privacy assessment of five large-scale datasets purchased from two different data brokers that collect accurate location traces without informed consent on a continuous basis.
Multiple experiments demonstrate the feasibility, magnitude and practical impact of diverse privacy attacks. Finally, we highlight realistic abuse scenarios, and propose solutions to mitigate these privacy concerns.

## Repository Content

The files in this repository are structured as follows:
*	**clustering.py**: contains code to perform the clustering of the location reports
*	**social_graph.py**: contains the social graph construction step
*	**social_graph_gpu.py**: contains the gpu version of the social graph construction step
*	**pipeline.py**: contains code to run all stages of the implementation in order
*	**environment.yml**: contains the minimal packages required to execute the implementation
*	**st_dbscan.py**: our modified version of st_dbscan which has support for chunking as described in the paper.

## Input dataset structure

Each dataset of location reports on which the implementation is performed must contain the following columns:
*	**DEVICE_ID**: The device identifier as an integer greater then zero.
*	**LAT**: The report latitude as a float.
*	**LON**: The report longitude as a float.
*	**EVENT_TIMESTAMP**: The report timestamp, indicating the offset in days from the dataset's start time, as a float.
