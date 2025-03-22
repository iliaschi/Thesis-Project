# Thesis

# Facial Emotion Recognition Using EfficientNet-B0: Implementation and Evaluation with Synthetic Data

## Abstract
This document outlines the implementation strategy for facial emotion recognition using EfficientNet-B0 with pre-trained weights and synthetic data integration. The methodology encompasses model architecture, synthetic data generation, and evaluation protocols.

## 1. Implementation Methodology

### 1.1 Model Architecture
The implementation utilizes the following architecture:
```
Input Layer (224×224×3)
    ↓
EfficientNet-B0 (with VGGFace2 weights)
    ↓
Feature Extraction Layer
    ↓
Emotion Classification (8 classes)
```

### 1.2 Pre-trained Model Integration
```python
# Model initialization sequence
model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
model.classifier = torch.nn.Identity()
model.load_state_dict(torch.load('path/to/state_vggface2_enet0_new.pt'))
```

### 1.3 Data Processing Pipeline
- Input image normalization
- Face detection and alignment
- Resolution standardization (224×224)
- Batch processing implementation

## 2. Synthetic Data Generation

### 2.1 Image Generation Protocol
- Resolution: 1024×1024 base images
- Grid configuration: 6×6 face layout
- Output: 36 individual faces per generation
- Total dataset size: 1,080 faces (30 credits)

### 2.2 Face Extraction Process
```python
# Processing sequence
1. Grid-based segmentation (170×170 per face)
2. Face detection validation
3. Quality assessment
4. Downsampling to 128×128
```

### 2.3 Emotion Categories
Eight distinct emotion classes:
- Anger
- Disgust
- Fear
- Happiness
- Neutral
- Sadness
- Surprise
- Contempt * in case of dataset support

## 3. Evaluation Framework

### 3.1 Testing Protocol
```markdown
Phase 1: Baseline Evaluation
- Load pre-trained weights
- Test on real dataset
- Record baseline metrics

Phase 2: Synthetic Integration
- Fine-tune with synthetic data
- Cross-validation
- Domain transfer assessment
```

### 3.2 Performance Metrics
```python
# Primary metrics
- Classification accuracy
- Per-class precision/recall
- F1-score
- Confusion matrix

# Secondary analysis
- Cross-domain performance
- Error pattern analysis
- Statistical significance tests
```

### 3.3 Validation Strategy
```markdown
1. Real Data Validation
   - Standard test set evaluation
   - Cross-dataset validation

2. Synthetic Data Validation
   - Internal consistency check
   - Cross-domain performance

3. Combined Analysis
   - Performance comparison
   - Domain gap assessment
```

## 4. Technical Implementation Details

### 4.1 Model Configuration 
#### To be subjected to changes 

```python
# Training parameters
learning_rate = 1e-4
epochs = 6
optimizer = RobustOptimizer(
    params,
    lr=learning_rate,
    rho=0.05
)
```

### 4.2 Data Organization
```
project_root/
├── data/
│   ├── real/
│   │   ├── train/
│   │   └── test/
│   └── synthetic/
│       ├── train/
│       └── test/
├── models/
│   └── weights/
└── results/
    └── metrics/
```

### 4.3 Implementation Requirements
- PyTorch framework
- CUDA-enabled GPU
- Required libraries:
  - timm
  - torch
  - PIL
  - numpy
  - sklearn

## 5. Expected Outcomes

### 5.1 Primary Deliverables
- Baseline performance metrics
- Synthetic data impact assessment
- Cross-domain analysis results

### 5.2 Analysis Components
- Per-emotion performance metrics
- Domain transfer effectiveness
- Model generalization capabilities

## 6. Future Considerations
- Extended synthetic data generation
- Advanced domain adaptation
- Multi-modal integration possibilities

## Promt Engineered with claude for using it in llms
# FER (Facial Emotion Recognition) Project Guide

## Project Context & Goals
Building emotion recognition system using:
- EfficientNet-B0 base model
- VGGFace2 pre-trained weights
- Limited synthetic data (30 image credits)
- 8 emotion classes

## Key Implementation Details
```python
# Core model structure
model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
model.classifier = torch.nn.Identity()
model.load_state_dict(torch.load('weights.pt'))

# Critical parameters
input_size = 224  # Model input
grid_size = 1024  # Synthetic images
face_size = 128   # After cropping
grid_layout = 6   # 6x6 faces per image
```

## Specific Questions Format
When asking about this project, specify:
1. Which stage: [weights loading/synthetic data/training/evaluation]
2. Current issue: [specific problem or goal]
3. Relevant constraints: [GPU memory/dataset size/etc]

## Example Questions
"Given this project structure, how would you:
- Load and verify the pre-trained weights are working correctly?
- Extract individual 128x128 faces from a 6x6 grid image?
- Implement efficient batch processing for training?
- Evaluate performance across real and synthetic data?"

## Technical Constraints
- Memory: Standard GPU (8-16GB VRAM)
- Data: 30 synthetic image credits
- Input: 224x224 for model
- Output: 8 emotion classes

## Expected Outputs
1. Working pre-trained model
2. Processed synthetic dataset
3. Performance metrics on both data types

When asking questions, please specify which component you're working with and any specific implementation challenges.



# Project Structure for FER Implementation

```
fer_project/
├── configs/
│   └── config.yaml           # Model and training configurations
│
├── data/
│   ├── real/                 # Real image datasets
│   │   ├── train/
│   │   └── test/
│   └── synthetic/            # Generated synthetic images
│       ├── raw/              # Original 1024x1024 grid images
│       └── processed/        # Extracted 128x128 faces
│
├── data/
│    ├── real/
│    │   ├── fer2013/
│    │   │   ├── train/
│    │   │   │   ├── angry/
│    │   │   │   ├── happy/
│    │   │   │   └── ...
│    │   │   └── test/
│    │   │       ├── angry/
│    │   │       ├── happy/
│    │   │       └── ...
│    │   │
│    │   └── affectnet/
│    │       ├── train/
│    │       │   ├── angry/
│    │       │   ├── happy/
│    │       │   └── ...
│    │       └── test/
│    │           ├── angry/
│    │           ├── happy/
│    │           └── ...
│    │
│    └── synthetic/
│          ├── raw_grids/
│          │   ├── train/
│          │   │   └── grid_images/
│          │   └── test/
│          │       └── grid_images/
│          │
│          └── processed/
│              ├── train/
│              │   ├── angry/
│              │   ├── happy/
│              │   └── ...
│              └── test/
│                  ├── angry/
│                  ├── happy/
│                  └── ...
│
├── models/
│   ├── __init__.py
│   ├── model.py             # EfficientNet model setup
│   └── weights/             # Pre-trained weights
│
├── utils/
│   ├── __init__.py
│   ├── data_processing.py   # Image processing utilities
│   └── evaluation.py        # Evaluation metrics
│
├── scripts/
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── generate_faces.py   # Synthetic data generation
│
├── requirements.txt
└── README.md
```

## Key Files Description

### Core Files
- `model.py`: EfficientNet implementation with pre-trained weights loading
- `data_processing.py`: Grid image processing and face extraction
- `train.py`: Main training loop
- `evaluate.py`: Performance evaluation

### Configuration
`config.yaml`: Contains model parameters:
```yaml
model:
  input_size: 224
  num_classes: 8
  weights_path: "models/weights/vggface2_weights.pt"

data:
  grid_size: 1024
  face_size: 128
  grid_layout: 6
```

This structure:
1. Separates concerns clearly
2. Keeps data organization simple
3. Maintains easy weight management
4. Provides straightforward script access


## Promt engineering
Using this as a base


### Some are smiling!?
'A single image arranged in a 6x6 grid, each cell containing a distinct male individual with diverse ethnicities, hair colors, and skin tones. The faces are front-facing with neutral expressions, photographed in a consistent, well-lit, studio environment, with uniform backgrounds. The layout must be equally spaced, and each face should be clearly separated from the others.'
Creates a 3x3 or a 5x5 dimension grid for some reasons.

'A single image ,containing 36 distinct male individuals with diverse ethnicities, hair colors, and skin tones. The faces are front-facing with neutral expressions, photographed in a consistent, well-lit, studio environment, with uniform backgrounds. The layout must be equally spaced, and each face should be clearly separated from the others.'
produces unequal results with chaotic rows and columns for flux schnell
Good results for flux dex

Adding the promt of no smiling brakes the photos

-A image arranged in a 6x6 grid, each cell containing a distinct male individual with diverse ethnicities, hair colors, and skin tones. The faces are front-facing with neutral expressions, photographed in a well lit environment. The layout must be equally spaced, and each face should be clearly separated from the others.
  -- Flux dev best, others bad.

Please generate 36 faces of men with neutral emotions, the faces are front-facing. The men should have a high diversity in gender, age, ethnicity, hair color, skin color, etc.
gives cartoon faces for flux + stable diff

Please generate 36 photo realistic faces of men with neutral emotions, the faces are front-facing. The men should have a high diversity in gender, age, ethnicity, hair color, skin color, etc.
gives cartoon faces for flux + stable diff


### Women
using best promts
- A image arranged in a 6x6 grid, each cell containing a distinct women individuals with diverse ethnicities, hair, hair colors, and skin tones. The faces are front-facing with angry expressions, photographed in a well lit environment. The layout must be equally spaced, and each face should be clearly separated from the others.

-angry showed hands in some images
A image arranged in a 6x6 grid, each cell containing a distinct women individuals with diverse ethnicities, hair, hair colors, and skin tones. The faces are front-facing with fear expressions, photographed in a well lit environment. The layout must be equally spaced, and each face should be clearly separated from the others.

A image arranged in a 6x6 grid, each cell containing a distinct women individuals showing distugst emotions. Each one with diverse ethnicities, hair, hair colors, and skin tones. The faces are front-facing with fear expressions, photographed in a well lit environment. The layout must be equally spaced, and each face should be clearly separated from the others.
Better result.
-Having the emotion at the beggining increases the intensity.


- Variations of intensity of emotions would change things
See and test how the varying intensity will work 

- Clustering of emotions to groups and seeing the accuracy then
One example would be positive vs negative emotions and seeing the accuracy then

- Accuracy over speficic categories instead of overall ones
do an emotion evaluation first instead of #


A photo of a movie of a 6x6 dimensions grid, so six images for height and six images for width, each image cell containing a distinct women individuals with diverse ethnicities, hair, hair colors, and skin tones. Faces are front-facing and show fear emotional expressions. The layout must be equally spaced, and each face should be clearly separated from the others.

Adding the movie in the promt increases the emotions showed.
A movie photo of a 6x6 dimensions grid of portraits, so six images for height and six images for width, each image cell containing a distinct women individuals with diverse ethnicities, hair, hair colors, and skin tones. Faces are front-facing and show fear emotional expressions and well light environment. The layout must be equally spaced, and each face should be full and clearly separated from the others.

When adding 'realistic' the models hallucinate.
This creates images more cartoonish. Maybe because it implies
that it is not realistic itself

DALLEE added 1 man in one of the images



A movie photo, of a 6x6 dimension grid of portraits with grey background, so six images for height and six images for width, each image cell containing a distinct women individuals with diverse ethnicities, hair, hair colors, and skin tones. Faces are front-facing and show high surprised emotional expressions and well light environment. The layout must be equally spaced, and each face should be realistic, full and clearly separated from the others.


new with age ranges 
A movie photo of a 6x6 dimensions grid of portraits, so six images for height and six images for width, each image cell containing a distinct women individuals with diverse ethnicities, hair, age ranges, and skin tones. Faces are front-facing and show fear emotional expressions and well light environment. The layout must be equally spaced, and each face should be full and clearly separated from the others.


A movie photo of a 6x6 dimensions grid of portraits, so six images for height and six images for width, each image cell containing a distinct women individuals with diverse ethnicities, hair, age ranges, and skin tones. Faces are front-facing and show fear emotional expressions and well light environment. The layout must be equally spaced, and each face should be full and clearly separated from the others.


best 
A grid of a 6x6 dimension professional headshot movie portaits, each image cell containing a distinct women individuals with diverse ethnicities, hair, age ranges and skin tones. Faces are front-facing and show extreme angry emotional expressions and well light environment. The layout must be equally spaced, and each face should be full shown and clearly separated from the others.

A grid of a 6x6 dimension professional headshot movie portaits, each image cell containing a distinct women individuals with diverse ethnicities, hair, age ranges and skin tones. Faces are front-facing and show extreme sad emotional expressions and well light environment. The layout must be equally spaced, and each face should be full shown and clearly separated from the others.



Best 

A photo grid of a 6x6 dimension headshot movie portaits, each image cell containing a distinct women individuals with diverse ethnicities, hair, age ranges and skin tones. Faces are front-facing and show neutral emotional expressions and well light environment. It is important that the layout must be equally spaced, and each face should be full shown and clearly separated from the others.



## Research Questions
ok now I am thinking so now i have real images from FER2013 and AffectNet and have the following research questions:

1) How effecientnetB0 and B2 when pretrained of real images perform on synthetic images
2) How the model when finetuned on synthetic performs of real images
3) What are the features that the models consider more important and when trained on real and what when trained on synthetic
4) What differences do the models find between the real and synthetic images

Performance of Pre-trained Models on Synthetic Images

1 Approach: Take EfficientNet-B0 and B2 pre-trained on real datasets (FER2013, AffectNet) and directly evaluate them on the synthetic dataset. Compare metrics (accuracy, F1-score) to gauge how well real-data knowledge transfers to artificial faces.
Implementation: Simply load the pre-trained weights, run inference on your synthetic dataset, and record relevant performance scores (per-class and overall).
Fine-tuning on Synthetic Data and Testing on Real Images

2 Approach: Fine-tune the pre-trained models with a portion of the synthetic dataset, then test on real FER2013 or AffectNet test sets. Observe whether performance improves or degrades on genuine faces.
Implementation: Freeze most layers initially and train only the final layers with the synthetic data. Gradually unfreeze additional layers if performance stabilizes. Evaluate on real test splits to assess generalization.
Feature Importance and Saliency Differences (Real vs. Synthetic)

3 Approach: Use interpretability methods (e.g., Grad-CAM, Layer-wise Relevance Propagation) to visualize which regions the model considers most significant for classification in both real and synthetic images.
Implementation: Generate saliency maps or heatmaps for each class/emotion and compare across domains. Look for consistent facial regions (like eyes, mouth) or anomalies unique to synthetic faces.
Model Perception Gaps Between Real and Synthetic

4 Approach: Analyze performance discrepancies and misclassifications, and check embeddings or internal representations (e.g., via t-SNE or PCA) to see how real vs. synthetic samples cluster.
Implementation: Extract penultimate-layer features for both datasets and visualize them. Observe if synthetic data forms distinct clusters, indicating domain-specific artifacts or gaps the model notices.


## Image transformation for tensorflow
```python
train_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
)
```

# Efficient net loading documentation 

### https://tfimm.readthedocs.io/en/latest/content/efficientnet.html

EfficientNet NoisyStudent models, trained via semi-supervised learning. These models correspond to the tf_... models in timm.

efficientnet_{b0, ..., b7}_ns

efficientnet_l2_ns_475

efficientnet_l2

wil use noisy student model for robustness, paper here https://arxiv.org/abs/1909.03172

good papers for eff net 
“Our largest model, EfficientNet-L2, needs to be trained for 3.5 days on a Cloud TPU v3 Pod, which has 2048 cores.”

Given the amount of compute used for this, I think that many people won’t be able to replicate and tweak it for further experimentation.

I hope you enjoyed reading this summary!
nice article : https://medium.com/@nainaakash012/self-training-with-noisy-student-f33640edbab2

tim version timm=0.4.5

References
https://arxiv.org/pdf/1911.04252.pdf
https://arxiv.org/pdf/1503.02531.pdf
https://arxiv.org/pdf/1909.13788.pdf
https://arxiv.org/pdf/1603.09382.pdf
https://arxiv.org/pdf/1909.13719.pdf


# 15/03/2025

## Idea : Train and Test on splits
Create splits that look something like: 
  - 100% women and 100% men
  - 100% women and 75% men
  - ...

Do a accuracy plot to see how it changes


See how the accuracy changes for different: 
  - Training splits based on full test
  - Testing splits based on full training

### Tasks 
  
  - Do the splits 
  - Do Testing and save results of all
  - Do Traing and Test, then save results of accuracy and metrics
  - Do plots of these 2 for accuracy and other metrics

## Heat Map of Faces of Correct and Incorrect predictions

See how it focuses on faces show the results on examples of the 2 classes correct and incorrect

See how the distribution is for these classes

## Confidense Levels distribution of Correct and Incorrect

To see how it works for correct and incorrect. Something like a reliability graph for each emotion on each Class and one reliability for All emotions together


# full finetuned gender splits at : C:\Users\ilias\Python\Thesis-Project\results\training_experiment_20250316_013917finetuned_real
# full synthetic gender splits at : C:\Users\ilias\Python\Thesis-Project\results\training_experiment_20250316_122332synthetic_on_vggface2

# Additional Metrics
ROC curves and AUC for each emotion (treating each as a binary classification)
Precision-Recall curves especially helpful for imbalanced classes
Top-K accuracy (is the correct emotion in the top 2 or 3 predictions?)
Confidence distribution plots for correct vs. incorrect predictions
Error analysis showing the most confidently misclassified examples
Confusion matrix normalization (both raw counts and percentages)

{
    "Angry": {
        "precision": 0.4518229166666667,
        "recall": 0.7995391705069125,
        "f1-score": 0.5773710482529119,
        "support": 434.0
    },
    "Contempt": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0.0
    },
    "Disgust": {
        "precision": 0.6651162790697674,
        "recall": 0.35049019607843135,
        "f1-score": 0.4590690208667737,
        "support": 408.0
    },
    "Fear": {
        "precision": 0.5,
        "recall": 0.6872964169381107,
        "f1-score": 0.578875171467764,
        "support": 307.0
    },
    "Happiness": {
        "precision": 0.9238625812441968,
        "recall": 0.8396624472573839,
        "f1-score": 0.8797524314765695,
        "support": 1185.0
    },
    "Neutral": {
        "precision": 0.7756653992395437,
        "recall": 0.6,
        "f1-score": 0.6766169154228856,
        "support": 680.0
    },
    "Sadness": {
        "precision": 0.647244094488189,
        "recall": 0.8598326359832636,
        "f1-score": 0.738544474393531,
        "support": 478.0
    },
    "Surprise": {
        "precision": 0.7923076923076923,
        "recall": 0.3130699088145897,
        "f1-score": 0.44880174291938996,
        "support": 329.0
    },
    "accuracy": 0.6851609526302015,
    "macro avg": {
        "precision": 0.594502370377007,
        "recall": 0.5562363469473364,
        "f1-score": 0.5448788505999782,
        "support": 3821.0
    },
    "weighted avg": {
        "precision": 0.7362575571540556,
        "recall": 0.6851609526302015,
        "f1-score": 0.6853912690562406,
        "support": 3821.0
    }
}


# Sources and links

Effnet pretrained source Andrey Savchenko : https://arxiv.org/pdf/2203.13436