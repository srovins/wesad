## Stress Detection from Multimodal Wearable Sensor Data using autoencoder latent features
### Dataset: WESAD 
https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29


Run scripts in the following order (can take a while) to prepare data and extract features from the latent layer of autoencoder model: </br>
1. Preprocess and merge subject data
</br>Command: <br>
python merge_subj_data.py
</br></br>
Input data path: 'data/WESAD/'</br>
Generates the following files in data folder:</br>
subj_merged_acc_w.pkl</br>
subj_merged_bvp_w.pkl</br>
subj_merged_eda_temp_w.pkl</br>
merged_chest_fltr.pkl</br></br>
2. Create autoencoder model and extract latent features
</br>Command: </br>
python extract_ae_latent_features.py
</br></br>
Input files:<br>
subj_merged_acc_w.pkl</br>
subj_merged_bvp_w.pkl</br>
subj_merged_eda_temp_w.pkl</br>
merged_chest_fltr.pkl</br>
  - Uses ae_feature_extractor.py to build and train autoencoder model and extract features. </br>
  - Save extracted features leaving one subject out into pickle files in features/train and features/test directories. The number in the filename indicates which subject was left out in each fold.</br></br>
3. SVM_classifier.ipynb - Build SVM classifier that uses latent features extracted by autoencoder for three class classification of WESAD dataset: neutral, stress, and ammusement. Results analysis also included.</br></br>
4. MLP_classifier.ipynb - Build MLP (Multi Layer Perceptron) classifier that uses latent features extracted by autoencoder for three class classification of WESAD dataset: neutral, stress, and ammusement. Results analysis also included.




