# India in the Haze: Country-Level PM2.5 Concentration Forecasting

**Developed by: Vishnu Prasad**

This project develops a deep learning model for forecasting PM2.5 concentrations across India using meteorological and emission data. The model employs advanced spatiotemporal techniques to predict air quality up to 16 hours ahead with high accuracy.

## Table of Contents
- [Project Overview](#project-overview)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Project Overview

Air pollution is a critical environmental challenge in India, with PM2.5 being one of the most harmful pollutants affecting public health. This project aims to develop an accurate forecasting system that can predict PM2.5 concentrations across the country to help authorities and citizens prepare for poor air quality events.

The model processes historical meteorological data and emission inventories to forecast PM2.5 levels 16 hours into the future at a spatial resolution covering the entire Indian subcontinent (140×124 grid).

## Methodology

Our approach combines several advanced techniques:

1. **Data Preprocessing**: 
   - Log transformation for skewed emission features
   - Z-score normalization for all variables
   - Episode detection using statistical methods

2. **Deep Learning Model**:
   - Convolutional LSTM encoder-decoder architecture
   - Multi-scale spatial encoding with dilated convolutions
   - Physics-informed wind warping for advection modeling
   - Attention mechanisms for spatial focus

3. **Training Strategy**:
   - Combined loss function optimizing multiple metrics
   - Episode-weighted training for extreme events
   - Teacher forcing with scheduled decay
   - Early stopping with patience mechanism

## Model Architecture

The forecasting model implements a sophisticated encoder-decoder architecture:

### Encoder
Processes 10-hour historical data:
- ConvLSTM layers for temporal processing
- Multi-scale dilated convolutions for spatial context
- Hidden state initialization for sequence processing

### Decoder
Autoregressively predicts 16 future time steps:
- Wind warping module using physics-based advection
- Episode detection from hidden states
- Spatial attention mechanism
- Residual connections for stable training

### Key Components

1. **ConvLSTM Cell**: Processes spatiotemporal sequences with gated recurrent units
2. **Spatial Encoder**: Multi-scale feature extraction using dilated convolutions
3. **Wind Warp Module**: Physics-informed transport modeling using wind vectors
4. **Episode Detector**: Identifies high pollution event regions
5. **Spatial Attention**: Focuses on relevant spatial regions
6. **Output Head**: Generates final PM2.5 predictions with uncertainty quantification

## Key Features

- **Multi-variable Input Processing**: Handles 15 meteorological and emission variables
- **Episode-aware Training**: Special weighting for high pollution events
- **Physics Integration**: Wind-based advection modeling
- **Adaptive Loss Function**: Combines SMAPE, spatial gradients, and correlation losses
- **Mixed Precision Training**: Efficient GPU utilization with AMP
- **Scalable Architecture**: Supports multi-GPU training

## Dataset

The model trains on data organized by months:
- **Periods**: April 2016, July 2016, October 2016, December 2016
- **Variables**: Meteorological (temperature, humidity, wind, radiation) and emissions (PM2.5, NH3, SO2, NOx, NMVOC, biomass burning)
- **Spatial Resolution**: 140×124 grid covering India
- **Temporal Resolution**: Hourly data

### Input Features
- `q2`: Specific humidity at 2m
- `t2`: Temperature at 2m
- `u10`, `v10`: Wind components at 10m
- `swdown`: Shortwave downward radiation
- `pblh`: Planetary boundary layer height
- `psfc`: Surface pressure
- `rain`: Precipitation
- Various emission species (PM2.5, NH3, SO2, NOx, NMVOC, biomass burning)

## Results

After training for 20 epochs:
- **Best Validation Loss**: 0.2179
- **Training Time**: ~10.33 hours on Tesla T4 GPU
- **Batch Size**: Effective batch size of 4 (2×2 with accumulation)
- **Prediction Horizon**: 16 hours ahead
- **Spatial Coverage**: Full Indian subcontinent (140×124 grid)

The model successfully generates predictions for 218 test samples with:
- Minimum predicted value: 0.0045 µg/m³
- Maximum predicted value: 500.00 µg/m³
- Mean predicted value: 35.31 µg/m³

## Requirements

All required dependencies are listed in `requirements.txt`. To install:

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare Data**: Ensure data is organized in the expected directory structure
2. **Configure Paths**: Update `DATA_ROOT` in the notebook to point to your data directory
3. **Train Model**: Execute the notebook cells sequentially
4. **Generate Predictions**: Run inference section to produce forecasts
5. **Output Format**: Predictions saved as NumPy array with shape (samples, height, width, time_steps)

## Project Structure

```
.
├── Meet_notebook.ipynb     # Main implementation notebook
├── requirements.txt        # Python dependencies
├── Meet_notebook.ipynb     # Main implementation/training notebook
├── Visualization.ipynb     # Visualization and analysis notebook
├── fadeconvolstm.ipynb     # Fade/ConvLSTM model notebook
├── best_fade_model.pt      # Trained model checkpoint
├── best_model_p2 (2).pt   # Trained Phase-2 model checkpoint
├── requirements.txt        # Python dependencies
├── license.md              # Project license
└── readme.md               # Project documentation
```

## Technical Details

### Hyperparameters
- Batch Size: 2 (effective 4 with accumulation)
- Epochs: 20
- Learning Rate: 3e-4 with cosine annealing
- Hidden Dimensions: 96
- Kernel Size: 3
- Number of Layers: 3
- History Window: 10 hours
- Forecast Horizon: 16 hours

### Loss Function Components
1. Episode-weighted SMAPE (50%)
2. Global SMAPE (25%)
3. Spatial gradient loss (10%)
4. Pearson correlation loss (15%)

### Hardware Requirements
- GPU with CUDA capability (tested on Tesla T4)
- Minimum 16GB VRAM recommended
- Python 3.7+

## Future Improvements

- Incorporate satellite data for enhanced spatial coverage
- Extend temporal horizon beyond 16 hours
- Integrate real-time data feeds for operational forecasting
- Develop ensemble methods for uncertainty quantification
- Implement transfer learning for other geographic regions

## Author

This project and its implementation are presented under the name **Vishnu Prasad**.

The technical implementation includes the ConvLSTM-based PM2.5 forecasting pipeline, trained model checkpoints, visualization workflow, and supporting notebooks included in this repository.

## Acknowledgments

This work was developed for the Environmental Hacks