## Stress Detection from Multimodal Wearable Sensor Data
### Dataset: WESAD 
https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29


Run scripts in the following order to prepare data and extract features from the latent layer of autoencoder model: </br>
</br>
1. Preprocess and merge subject data</br>
<br>
Input data path: 'data/WESAD/'</br>
Generates the following files in data folder:</br>
subj_merged_acc_w.pkl</br>
subj_merged_bvp_w.pkl</br>
subj_merged_eda_temp_w.pkl</br>
merged_chest_fltr.pkl</br>
<br> 
Command: <br>
python merge_subj_data.py
