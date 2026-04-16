# Overview

The main objective is to fine-tune the SAM model with images from the medical domain.

# Relevant context:

- Assignment instructions: /Users/da625117/Documents/Courses/CAP5516_MedicalImageComputing/hw3/Assignment-3.pdf
    * This PDF file contains the instructions for assignment. We should only focus on the implementation of the training and evaluations scripts. Downloading the data and generating the report is out of scope.

- Dataset paper: /Users/da625117/Documents/Courses/CAP5516_MedicalImageComputing/hw3/2308.01760v1.pdf
    * This PDF contains the paper for the dataset that we are going to use for training. This contains all the details describing the data and annotations. We need to adhere to the evaluation protocols closely

- Dataset repo: /Users/da625117/Documents/Courses/CAP5516_MedicalImageComputing/hw3/NuInsSeg
    * This folder contains code provided by the author of the paper regarding the dataset. Use it only for reference.

- Dataset: /Users/da625117/Documents/Courses/CAP5516_MedicalImageComputing/hw3/archive
    * This folder contains the actual dataset that we will use for training. You need to identify which are the relevant masks for training

- Finetune SAM repo: /Users/da625117/Documents/Courses/CAP5516_MedicalImageComputing/hw3/finetune-SAM
    * This repo is suggested in the assignment to fine-tune SAM. We will use the MobileSAM with LoRA fine-tuning strategy


# Details:

- You must create the training and evaluations scripts in order to fine-tune SAM on the medical imaging dataset. 
- You can reuse any part of the existing repos in order to do so.
- We should be able to fine-tune the model in a machine with 24 GB GPU, and 32 GB of RAM, with the option to scale to 48 or 80 GB GPU and 72 GB RAM. 
- You should document everything including instructions about how to run the fine-tuning. 
- Make the necessary dataloaders, training scripts, evaluation scripts, and a script that connects everything together. Ideally, we want to train for a fixed amount of iterations and run the evaluations every N iterations to choose the best checkpoint. After training is finished, we run a final test evaluation.
- The metrics at every eval step must be saved to disk, as well as the final metrics with the best checkpoint.
- We also want to save a few predicted masks and make plots comparing the ground truth mask and the predicted masks. For example, we can choose up to 5 examples from each class of organ and plot the original image, ground truth mask, and predicted mask.
- In general make sure all the code is clean and easy to debug. Try to keep it straightforward and don't overcomplicated. Assume all the data is clean and working well.
- I'm not sure which masks we should choose for training, as there are several options. You should choose the one that best matches the assignment instructions and the paper.
- Try to save all the necessary data and training curves so that I can later make a good report. 
- Don't forget to log all your design choices and instructions to run the scripts into some md file in the project root.
- If we need external libraries that are not installed make a list so that I can install them before running the scripts