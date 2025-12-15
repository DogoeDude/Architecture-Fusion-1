# Graphics and Visual Computing – Mini Case Study Guidelines (Final Requirement)

## 1. Overview
    For your final requirement, you will complete a Mini Case Study that applies concepts from Graphics and Visual Computing to a deep learning–based computer vision task. The goal is to demonstrate your understanding of image processing, feature representation, visualization, and the use of neural network architectures.

    This is a short, focused, and practical project. You will be given a little over one week to complete it.

## 2. Project Theme
    Choose one computer vision task:
        Image Classification
        Object Detection
        Semantic Segmentation
        Image Generation / Style Transfer
        Super-Resolution / Image Enhancement
    You may use any dataset (public or self-created), except datasets from kaggle (there is a big tendency that these datasets are already well explored).

## 3. Core Requirement: Architecture Fusion
    Your model must demonstrate fusion or combination of two or more deep learning components, such as:
        Fusing CNN + Transformer
        Fusing two CNN backbones (e.g., ResNet + MobileNet features)
        Using GAN + Autoencoder
        Using U-Net + Attention Module
        Using feature concatenation from two pretrained models
        Using two feature extractors merged before classification
    The fusion does not need to be overly complex—what matters is you can explain:
        What you fused
        Why you fused them
        What improvement or behavior you expected

## 4. Deliverables
    A. Mini Report (<5 pages) in LNCS format
        Your report should contain:
        Title
        Introduction
            Problem being solved
            Why it is relevant
        Dataset Description
            Source, size, sample images
        Methodology
            Architectures used
            Explanation of the fusion strategy
            Preprocessing and training details
        Results & Visualizations
            Accuracy/IoU/loss curves
            Sample predictions (correct and incorrect)
            Visual explanation of the architecture (diagram or image)
        Discussion
            What your fusion contributed
            What worked, what didn’t
        Conclusion
        LNCS format - https://www.overleaf.com/latex/templates/springer-lecture-notes-in-computer-science/kzwwpvhwnvfj.pdf
    B. Code Notebook (Jupyter or Google Colab)
        Should run end-to-end
        Must include comments and visual outputs
        Clear function organization
        Create a dedicated github repository for this case study and add me as collaborator
        JhunBrian | andam.jhunbrian@gmail.com

## 5. Evaluation Criteria
    Criterion	Weight
    Clarity of Problem + Dataset Choice	15%
    Correctness of Method / Implementation	25%
    Fusion of Deep Learning Components	25%
    Quality of Visualizations (graphs, images, architecture)	15%
    Results + Interpretation	20%
6. Allowed Tools
    PyTorch
    OpenCV
    FastAI
    Google Colab
    Any pretrained model (allowed and encouraged)
    You may use torch frameworks like pytorch lightning for your training.

## 6. Groupings
    Maximum of 3 members, if you are confident enough to do this alone, iz good as well...

## 7. Tips
    Choose a small, fast dataset.
    Use transfer learning to save time.
    Focus on visual clarity—show images, comparisons, and diagrams.
    Keep the fusion simple but meaningful.
    Prioritize working code over complex theory.
    For every step and procedure you do, make sure it is data-driven