# Handwritten Digit Recognition using Neural Networks

This project implements a handwritten digit recognition system using a Multi-Layer Perceptron (MLP) neural network in PyTorch. The system is designed to classify handwritten digits from 0 to 9 with high accuracy.

## Project Structure

### Core Model Files

#### `MyModel.py`
The main model implementation containing:
- **MLP_net class**: A 4-layer neural network with dropout regularization
  - Input layer: 784 neurons (28x28 flattened images)
  - Hidden layers: 512 → 256 → 128 neurons
  - Output layer: 10 neurons (for digits 0-9)
  - Dropout rate: 0.3 between layers
- **MyModel function**: Loads pre-trained weights and performs inference on single images
- Uses ReLU activation function and runs on CPU for inference

#### `final_model.py`
Complete training pipeline including:
- Data loading and preprocessing with torchvision transforms
- Model training with Adam optimizer and CrossEntropyLoss
- Training loop with validation monitoring
- Performance evaluation with accuracy metrics and confusion matrices
- Model saving functionality
- Visualization of training/validation loss over epochs

#### `test_data_load.py`
Testing and evaluation script that:
- Loads the trained model weights
- Tests model performance on a test dataset
- Calculates accuracy using custom function
- Generates confusion matrix visualization
- Provides detailed performance metrics

### Data Processing Scripts

#### `DataPreprocessing.py`
Preprocessing pipeline for input images:
- Converts images to grayscale
- Resizes images to 28x28 pixels
- Normalizes pixel values to range [-1, 1]
- Converts numpy arrays to PyTorch tensors
- Flattens images for MLP input
- Handles different input formats (uint8 conversion)

#### `split_data.py`
Data splitting utility:
- Splits dataset into training (80%), validation (10%), and test (10%) sets
- Uses stratified splitting to maintain class distribution
- Copies images to respective directories
- Ensures reproducible splits with random seed

#### `digit_organizer.py`
Data organization script:
- Sorts images into digit-specific folders (0-9)
- Maps English digit names to numeric folders
- Handles image file extensions (.png, .jpg, .jpeg)
- Creates directory structure for organized dataset

#### `data_augmentation.py`
Data augmentation pipeline:
- Converts all images to grayscale
- Applies three augmentation techniques:
  - **Rotation**: Random rotation (-40° to +40°)
  - **Mirroring**: Horizontal flip
  - **Noise**: Gaussian noise addition
- Triples the dataset size for better generalization

### Utility Scripts

#### `len_of_dir.py`
Dataset analysis tool:
- Counts occurrences of each digit (0-9) in the dataset
- Generates bar chart visualization of class distribution
- Helps identify class imbalance issues
- Provides total dataset size statistics

#### `Main.py`
Main evaluation framework:
- Loads test data from CSV references
- Applies preprocessing pipeline
- Runs model inference on test images
- Builds confusion matrix for performance evaluation
- Calculates F1 scores using `GetScoreOcr`
- Handles class mapping (converts class 10 to 0)

#### `GetScoreOcr.py`
Performance evaluation utility:
- Calculates partial F1 scores for each digit class
- Computes overall F1 score across all classes
- Uses confusion matrix for precision/recall calculations
- Provides standardized evaluation metrics

## Model Architecture

The neural network uses the following architecture:
```
Input (784) → FC1 (512) → ReLU → Dropout(0.3) → 
FC2 (256) → ReLU → Dropout(0.3) → 
FC3 (128) → ReLU → Dropout(0.3) → 
FC4 (10) → Output
```

## Data Pipeline

1. **Raw Data** → Images of handwritten digits
2. **Organization** → Sort into digit-specific folders (0-9)
3. **Augmentation** → Apply rotation, mirroring, and noise
4. **Splitting** → Divide into train/validation/test sets
5. **Preprocessing** → Convert to grayscale, resize, normalize
6. **Training** → Train MLP model with validation monitoring
7. **Evaluation** → Test performance and generate metrics

## Key Features

- **Robust Preprocessing**: Handles various image formats and sizes
- **Data Augmentation**: Improves model generalization
- **Dropout Regularization**: Prevents overfitting
- **Comprehensive Evaluation**: F1 scores and confusion matrices
- **Modular Design**: Separate scripts for different pipeline stages
- **Visualization**: Loss curves and performance metrics

## Usage

1. Organize your digit images using `rozrazovac.py`
2. Split data into train/val/test sets with `split_data.py`
3. Apply data augmentation using `data_augmentation.py`
4. Train the model with `final_model.py`
5. Evaluate performance using `test_data_load.py` or `Main.py`

## Performance

The model achieves high accuracy on handwritten digit recognition through:
- Proper data preprocessing and augmentation
- Balanced neural network architecture
- Regularization techniques
- Comprehensive evaluation methodology

## Dependencies

- PyTorch
- torchvision
- OpenCV (cv2)
- PIL (Pillow)
- scikit-learn
- matplotlib
- numpy
- pandas

## Model Weights

The trained model weights are saved as `mymodel_3.pth` and can be loaded for inference using the `MyModel` function.

