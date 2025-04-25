# DSEM6105
# Concept Bottleneck Models for Bird Classification

![Bird Classification Example](https://github.com/user/concept-bottleneck-birds/raw/main/images/example_predictions.png)

## Overview

This project implements Concept Bottleneck Models (CBMs) for fine-grained bird species classification using the CUB-200-2011 dataset. Unlike traditional "black box" models, CBMs provide interpretability by forcing predictions to pass through human-understandable concepts, allowing for explanation and intervention in the prediction process.

## Features

- **Interpretable Classification**: Models explain their decisions using human-understandable concepts
- **Interactive Prediction**: Users can correct model misconceptions to improve results
- **Multiple Architectures**: Implementation of standard, sequential, and independent models
- **Visualization Tools**: Generation of prediction visualizations with confidence scores
- **Intervention Analysis**: Evaluation of how concept corrections improve performance

## Dataset

This project uses the Caltech-UCSD Birds-200-2011 (CUB) dataset:
- 11,788 images of 200 bird species
- 312 binary attribute annotations (we use a subset of 112)
- Split into train (4,796 samples), validation (1,198 samples), and test (5,794 samples) sets

The preprocessed data is stored in pickle files:
- `train.pkl`: Training data
- `val.pkl`: Validation data
- `test.pkl`: Test data

Each pickle file contains a list of dictionaries with:
- `id`: Sample identifier
- `img_path`: Path to the bird image
- `class_label`: Bird species label (0-199)
- `attributes`: Binary vector of 112 attributes

## Model Architectures

### Standard Model (Baseline)
A traditional end-to-end model that directly maps images to class predictions:
```
Image → ResNet50 → Class Prediction
```

### Concept Bottleneck Model (CBM)
Forces predictions to pass through interpretable concepts:
```
Image → ResNet50 → Concept Predictions → Class Prediction
```

### Independent Model
Predicts concepts and classes from shared features:
```
                   ┌→ Concept Predictions
Image → ResNet50 →┤
                   └→ Class Prediction
```

### Sequential Model
Trained in two phases for concept prediction then class prediction:
```
Phase 1: Image → ResNet50 → Concept Predictions
Phase 2: [Freeze] → Concept Predictions → Class Prediction
```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/user/concept-bottleneck-birds.git
cd concept-bottleneck-birds
pip install -r requirements.txt
```

## Data Preparation

Download and prepare the CUB dataset:

```bash
# Download CUB dataset
python download_data.py

# Preprocess data
python preprocess_data.py
```

## Usage

### Training

Train different model architectures:

```bash
# Train standard model
python concept_bottleneck_experiment.py --model_type standard

# Train concept bottleneck model
python concept_bottleneck_experiment.py --model_type cbm

# Train sequential model
python concept_bottleneck_experiment.py --model_type sequential

# Train independent model
python concept_bottleneck_experiment.py --model_type independent
```

### Evaluation

Evaluate models and visualize predictions:

```bash
# Evaluate standard model
python test_model.py --test_file test.pkl --model_path results/standard_model.pth --model_type standard --save_images

# Evaluate CBM with interventions
python test_model.py --test_file test.pkl --model_path results/cbm_model.pth --model_type cbm --test_interventions --save_images
```

## Results

Our experiments show:
- Standard Model achieves ~73% test accuracy
- Concept Bottleneck Model achieves ~71% accuracy with interpretability
- With concept interventions, accuracy improves to ~78%

## Example Output

For each test image, the model outputs:
- True bird species
- Top 5 predicted species with confidence scores
- Visualization of the image with prediction results

Example:
```
Example 1:
True bird species: 034.Bird_Species_34
Top 5 predicted species:
1. 034.Bird_Species_34: 0.1471
2. 035.Bird_Species_35: 0.0622
3. 037.Bird_Species_37: 0.0512
4. 033.Bird_Species_33: 0.0489
5. 036.Bird_Species_36: 0.0456
```

## Intervention

The key advantage of CBMs is the ability to intervene on concept predictions:

1. Model predicts attributes (concepts) from an image
2. User reviews and corrects any erroneous concept predictions
3. Model uses corrected concepts to update class prediction

Our analysis shows that correcting just ~10% of concept errors can improve classification accuracy by 5-7%.

## Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision
- NumPy
- matplotlib
- PIL
- tqdm

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code for your research, please cite our project:

```
@misc{concept_bottleneck_birds,
  author = {Your Name},
  title = {Concept Bottleneck Models for Bird Classification},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/user/concept-bottleneck-birds}
}
```

## Acknowledgments

- This project is based on the "Concept Bottleneck Models" paper by Koh et al. (ICML 2020)
- We use the CUB-200-2011 dataset provided by Caltech-UCSD
