# Earthquake-Magnitude-Estimation
Designed and implemented a high-performance earthquake magnitude prediction system leveraging machine learning, deep learning, and GPU acceleration for computational optimization.
Collected large-scale historical seismic data from the United States Geological Survey (USGS) and performed structured preprocessing including time-index sorting, duplicate removal, and feature normalization using MinMax scaling.
Engineered time-series sequences for supervised learning to model temporal dependencies in seismic magnitude prediction.
Implemented and optimized:
Random Forest Regressor
XGBoost for gradient-boosted tree optimization
GRU-based Recurrent Neural Network in PyTorch
Designed and trained the GRU model with configurable execution on:
CPU
CUDA-enabled GPU
Leveraged CUDA acceleration to significantly reduce training time and analyzed computational performance improvements (CPU vs GPU speedup factor).
Evaluated model performance using:
RMSE (Root Mean Square Error)
MAE (Mean Absolute Error)
Training execution time benchmarking
Conducted comparative analysis of:
Traditional ML vs Deep Learning models
CPU vs GPU computational efficiency
Accuracy vs compute trade-offs
Demonstrated practical application of High-Performance Computing (HPC) principles including parallel processing, workload optimization, and hardware-aware model training.
Developed a reproducible experimental pipeline for performance benchmarking in heterogeneous compute environments.
