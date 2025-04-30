1. final_UI.py is the main file to run for the UI, and a picture of UI is below:

<img width="750" alt="ui_picture" src="https://github.com/user-attachments/assets/a8f84da4-8496-47a9-91f6-90d4d9272639" />

2. clinical_targets.csv, pathomics.csv, radiomics.csv and transcriptomics.csv are original data files, while final_merged_rfe.csv is the final data file after data preprocessing. final_glossary.csv contains description for each feature inside final_merged_rfe.csv

3. data_preprocessing.py contains all methods for data preprocessing, while final_merged_rfe.csv is the final data file after data preprocessing. final_glossary.csv contains description for each feature inside final_merged_rfe.csv

4. get_synthetic_data.py contains all functions for generating synthetic data by SMOTE 

5. model.py contains all code for MLP, both training and evaluating
 
6. run_shap.py contains all code for calculating SHAP values

7. improved.py contains all code for invoking perplexity model 
