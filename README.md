# Lung-Cancer-Image-Classification-with-PyTorch



##  Problem Description

The dataset includes chest X-ray or CT scan images categorized into 4 lung cancer classes:

- Normal
- Large Cell Carcinoma
- Adenocarcinoma
- Squamous Cell Carcinoma

Each class is placed in a separate folder. The task is to build a CNN that classifies input images into these 4 categories.

---

##  Model Goals and Tasks

1. **Train-Test Split**
   - Randomly split dataset into training and testing sets.

2. **Data Augmentation**
   Apply the following augmentations randomly with 20% probability:
   - Random resized crop
   - Vertical and horizontal flips
   - Contrast adjustment
   - Random rotation (±30 degrees)

3. **Regularization**
   - Apply dropout and weight decay to prevent overfitting.

4. **Hyperparameter Tuning**
   - Use tools like `wandb` or `comet.ml` (optional) to track:
     - Learning rate
     - Batch size
     - Dropout rate
     - Number and size of filters

5. **Pre-trained Model Comparison**
   - Load a pre-trained model from torchvision (e.g. ResNet)
   - Freeze feature extractor layers and retrain classifier head
   - Adjust input preprocessing as required

6. **Visualization and Evaluation**
   - Plot training/validation loss and accuracy
   - Report final accuracy on the test set
   - Confusion matrix for class-wise evaluation

---

##  Technologies Used

- Python 3.8+
- PyTorch
- torchvision
- matplotlib, seaborn
- PIL
- wandb (optional)

Install requirements using:

```bash
pip install torch torchvision matplotlib seaborn wandb
