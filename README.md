# Federated Trees

A Python implementation of federated decision tree learning, corresponding to the approach described in **“Federated Trees: ….”** (by Nisal Hemadasa et al.).  
This repository provides code, examples, and utilities to build, train, and evaluate federated tree models in a distributed manner.

---

## Paper / Reference

Please cite the original work as:
 
<pre>@INPROCEEDINGS{11119244,
  author={Hemadasa, Nisal and Kaaser, Dominik and Schulte, Stefan},
  booktitle={2025 10th International Conference on Fog and Mobile Edge Computing (FMEC)}, 
  title={Hierarchical Bidirectional Aggregation for Federated Learning Under Concept Drift}, 
  year={2025},
  volume={},
  number={},
  pages={133-140},
  keywords={Adaptation models;Accuracy;Federated learning;Network topology;Scalability;Concept drift;Vegetation;Topology;Servers;Resilience;Machine Learning;Hierarchical Federated Learning;Concept Drift;Bidirectional Aggregation},
  doi={10.1109/FMEC65595.2025.11119244}}</pre>

---

## Features

- Training decision tree / ensemble models in a federated setting (without centralized data)  
- Communication protocols for aggregation and secure information exchange  
- Support for drift / evolving data over time  
- Experimental notebooks for simulating drift, testing strategies  
- Logging, plotting, and utility modules  

## Repository Structure

<pre>. 
├── data/ # Datasets or loaders 
├── drift_concepts/ # Concept drift modules 
├── federated_network/ # Server–client communication 
├── logs/ # Training logs 
├── models/ # Tree model implementations 
├── plots/ # Plotting utilities and figures 
├── strategy/ # Federated strategies 
├── constants.py # Global configs 
├── main.py # Entry point 
├── *.ipynb # Example notebooks 
├── LICENSE # License (Apache 2.0) 
├── README.md # This file 
└── backlog.txt # To-do / notes </pre>