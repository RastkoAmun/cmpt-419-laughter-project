# Laughter Classification with Convolutional Neural Network

This project explores the classification of different types of human laughter using a Convolutional Neural Network (CNN). Our goal is to detect and classify four categories of laughter — **chuckle**, **belly laugh**, **baby laughter**, and **evil laugh** — using audio clips and Mel-frequency cepstral coefficients (MFCCs).

---

## 🛠️ Getting Started

To set up and run the project locally:

```bash
git clone https://github.com/RastkoAmun/cmpt-419-laughter-project
OR
git clone git@github.com:RastkoAmun/cmpt-419-laughter-project.git

cd cmpt-419-laughter-project 

python3.12 -m venv venv
source venv/bin/activate # on MAC
pip install -r requirements.txt
```

## 📁 Project Structure

```bash
|-- src
|   |-- data
|   |   |-- training-dataset   # folder with all data
|   |   |-- labels.csv         # our manual labels for the data
|   |-- helpers
|   |   |-- helpers.py         # Path management and audio helper functions
|   |   |-- utils.py           # Utility functions 
|   |-- model_sandbox.ipynb    # Notebook for early model experiments
|   |-- model.ipynb            # Final model training and evaluation notebook
|
|-- .gitignore                 # Git ignore rules
|-- requirements.txt           # Project dependencies
|-- README.md                  # Project overview and setup instructions
```

## 📝 Self-Evaluation

We completed all the goals outlined in our original project proposal. We successfully collected and labeled a dataset of laughter clips, performed MFCC feature extraction using librosa, and trained a Convolutional Neural Network (CNN) to classify different types of laughter.

One additional improvement we made beyond the proposal was the inclusion of a fourth laughter category: evil laugh. This class added complexity and variety to the dataset and allowed us to test the model’s ability to generalize across more theatrical and unpredictable vocal patterns.

We are satisfied with the final outcome and believe the project meets all the criteria we initially set, along with offering an extra challenge we tackled successfully. 


## 📦 Dependencies

This project uses the following libraries:
- PyTorch
- Seaborn
- Pandas
- NumPy
- librosa
- matplotlib
- sklearn
- pathlib