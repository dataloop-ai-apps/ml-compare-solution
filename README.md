# ML Comparison Solution

Welcome to the ML Comparison Solution repository, an application designed to showcase the capabilities of the Dataloop platform in managing and evaluating machine learning models. This solution is available under the `Datasets` tab in the Dataloop Marketplace and leverages the V2 Plant Seedlings Dataset from Kaggle.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Components](#components)
4. [App Code Overview](#app-code-overview)
5. [Configuration Overview](#configuration-overview)
6. [Additional Information](#additional-information)

## Overview

The ML Comparison Solution utilizes a machine learning model, specifically ResNet, to analyze two different versions of the V2 Plant Seedlings Dataset: one annotated and one unannotated.

### Key Features

- **Dataset Management**: Seamlessly handle annotated and unannotated datasets.
- **Model Training and Evaluation**: Train and evaluate ResNet models with precision.
- **Performance Metrics**: Access detailed metrics including loss and accuracy.

## Installation

To install the ML Comparison Solution app:

1. Navigate to the **Marketplace** on the Dataloop platform.
2. Select the **Datasets** tab.
3. Search for the "ML Comparison Solution" app.
4. Click on **Install** to add the solution to your workspace.
5. Once installed, the datasets will appear under your **Data** section and the models will be available in the **Models** section.
6. You can now look at the images and annotations in the datasets created and compare models by selecting the models and then selecting the **Compare Models** button.

## Components

The solution comprises several key components:

- **Datasets**: 
  - **Annotated**: V2 Plant Seedlings Dataset with annotations.
  - **Unannotated**: V2 Plant Seedlings Dataset without annotations.
- **Models**: Three trained ResNet models, each with detailed training metrics and precision-recall figures.
- **Recipe and Ontology**: Comprehensive labeling for dataset management.

## App Code Overview

### Dataset Adapter

The `Loader` class in `loader.py` is the core of the code. It includes methods to load both annotated and unannotated datasets, as well as to clone and evaluate pre-trained models. Key methods include:

- `upload_dataset`: Create an ontology and upload the dataset to the Dataloop platform.
- `upload_data`: Uploads the dataset to the Dataloop platform (by calling the upload_dataset function).
- `load_unannotated`: Uploads the unannotated datasets (by calling the upload_data function).
- `clone_models`: Clones an existing Resnet model and create 3 new models inside the projects (with names agri-classification-v1, agri-classification-v2, agri-classification-v3). The function is used to replicate an evaluation of a model (by getting pre-computed metrics and plots) done on a specific dataset and to show the model comparison possibilities in the Dataloop plaform.
- `load_annotated`: Uploads the annotated datasets (by calling the upload_data function), retrieve models metrics and calls the clone_models function.


## Configuration Overview

The `dataloop.json` file serves as the core configuration for the solution. It defines various parameters, including app details, compute resources, modules, and datasets.

### Key Components

#### 1. App Information
- **Display Name**: Name as it will be displayed in the Dataloop platform.
- **Name**: Name of the dpk behind the app.
- **Description**: App description.
- **Scope**: App scope can be public (for all users) or project (for a specific project).
- **Version**: App version.
- **Codebase**: Codebase details including type (git), url and tag.

#### 2. Attributes
Defines the app’s characteristics:
- **Category**: Specifies the type of app (Dataset, Model, etc.).
- **Hub**: Indicates the marketplace hub where the app is published.
- **Media Type**: Defines supported file types (here Image).
- **Annotation Type**:  Specifies the annotation approach (here Classification).
- **Computer Vision**: Defines the type of computer vision task (here Classification).
- **License**: `CC BY-SA 4.0`

#### 3. Components
##### 3.1 - Compute Configurations
Specifies resources and execution settings:
- **Name**: Compute configuration name.
- **Runtime**: Defines execution parameters:
  - **Pod Type**: Machine type for running the app.
  - **Concurrency**: Maximum concurrent tasks (set to `10`).
  - **Autoscaler**:
    - **Min Replicas**: Minimum running instances (`0`—scales down to zero when idle).
    - **Max Replicas**: Maximum running instances (`1`—only one active instance at a time).
    - **Queue Length**: Triggers new replicas when pending tasks exceed this limit.

##### 3.2 - Modules
Modules define distinct functionalities within the app:
- **Name**: Module name.
- **Entry Point**: The Python script executed on launch (e.g., `loader.py`).
- **Class Name**: `Loader` (class defined in `loader.py`).
- **Compute Config**: Specifies the compute setup for the module.
- **Description**: Brief summary of the module’s purpose.
- **Init Inputs**: Inputs required to initialize the module.
- **Functions**: Defines the functions the module can use.

##### 3.3 - Datasets
The app utilizes two versions of the **V2 Plant Seedlings Dataset** from Kaggle:
- **Annotated version**:  
  `https://storage.googleapis.com/model-mgmt-snapshots/datasets-agriculture/annotated.zip`
- **Unannotated version**:  
  `https://storage.googleapis.com/model-mgmt-snapshots/datasets-agriculture/unannotated.zip`

#### 4. Dependencies
- Includes a simplified `dataloop.json` configuration for a **ResNet model adapter**, focusing on relevant functions.
- For a full implementation, refer to the **ResNet app** in the [Dataloop Apps GitHub 'torch-models' repository](https://github.com/dataloop-ai-apps/torch-models).


## Additional Information

For more details on the V2 Plant Seedlings Dataset, visit the [Kaggle dataset page](https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset).