# Transfer Learning vs. Fine-Tuning: A Comparative Analysis

This repository contains the scripts and presentation materials for our analysis of the key differences between transfer learning and fine-tuning using the MTCNN (Multi-task Cascaded Convolutional Networks) model for facial recognition tasks.

## Project Overview

Our project aims to evaluate the pre-trained MTCNN model against a fine-tuned version with a custom P-Net (Proposal Network) layer. The study highlights trade-offs in accuracy, runtime, and robustness, offering practical insights into choosing between these approaches for domain-specific tasks.

## Key Concepts
### Transfer-Learning

<ol>- Utilizes the pre-trained weights of the entire MTCNN model.</ol>
<ol>- Suitable for quick deployments without significant modifications.</ol>
<ol>- Requires minimal computational resources.</ol>

### Fine-Tuning

<ol>- Involves retraining specific layers, such as the P-Net, while freezing others.</ol>
<ol>- Adapts the model to the target domain for better specificity.</ol>
<ol>- Demands more computational power and debugging efforts.</ol>

## Methodology
### Dataset

<ol>- <b>Images</b>: 4,000 images with 480x360 resolution.</ol>
<ol>- <b>Annotations</b>: Bounding boxes for detected faces, including multiple faces per image.</ol>

### Steps for Transfer Learning

<ol>1. Load pre-trained MTCNN model.</ol>
<ol>2. Predict bounding boxes and confidence scores.</ol>
<ol>3. Evaluate using metrics such as precision, recall, and average precision on a 200-image test sample.</ol>

### Steps for Fine-Tuning

<ol>1. <b>Preproccessing</b>: Rescale images and create bounding box annotations.</ol>
<ol>2. <b>Layers Freezing</b>: Freeze R-Net and O-Net layers to retain pre-trained weights.</ol>
<ol>3. <b>P-Net Training</b>:
  <ol>- Add custom convolutional layers and activation functions.</ol>
  <ol>- Define classification and bounding box regression losses.</ol>
  <ol>- Train using a custom training loop.</ol>
</ol>

### Testing & Evaluation

<ol>- Compare predicted bounding boxes to ground truth using Intersection over Union (IoU).</ol>
<ol>- Compute precision, recall, and generate a Precision-Recall Curve.</ol>

## Results
### Expected Outcomes
| Pre-trained MTCNN         | Fine-tuned MTCNN            |
|----------------------------|-----------------------------|
| Faster execution           | Improved specificity       |
| Better general performance | Adapted to the dataset     |

<img src = 'https://github.com/leviaici/tfvsft/blob/main/404_trained_200.png' align="right" width = 250>
<img src = 'https://github.com/leviaici/tfvsft/blob/main/pretrained_200.png' align="right" width = 250>

### Real Outcomes

<ol>- <b>Speed</b>: Pre-trained model is 6-7x faster than the fine-tuned version.</ol>
<ol>- <b>Ease of Use</b>: Fine-tuning required extensive debugging.</ol>
<ol>- <b>Performance</b>: Fine-tuned model achieved a 4.1% improvement in performance.</ol>

## Conclusion

The pre-trained MTCNN model excels in speed, making it ideal for time-sensitive applications, while the fine-tuned model offers slightly better performance for domain-specific datasets. The choice between these approaches depends on the trade-offs between speed and accuracy required by your use case.

## Notes

Please note that this homework assignmenta is a part of our academic coursework at the University of Bucharest. Feel free to explore the code, documentation, and solutions to gain insights into the exciting world of robotics and computer science.

## Disclaimer
This repository is for educational purposes, and you are encouraged to learn from it. However, please avoid direct copying of code for your own coursework.
