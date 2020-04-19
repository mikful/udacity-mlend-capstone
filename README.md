# Udacity Machine Learning Engineer Nanodegree - Capstone Project
## Multi-Label Auto-Tagging of Audio Files Using fastai2 Audio



## Dependencies

The EDA and model development were undertaken within a PyTorch 1.4 kernel within GCP.

 For the model development a P100 GPU AI Notebook Instance was used.

The dependencies summarised are required to run the notebooks, detailed steps to do so are given directly within the relevant notebooks found within the `nbs_final` folder.

## EDA

The following libraries were used as shown within the `eda.ipynb` notebook:

* `numpy`
* `pandas`
* `torchaudio`

* `matplotlib.pyplot`

* `IPython.display`

* `librosa`



## Model Development 

The following libraries are required to run the notebook  `model_dev_and_test.ipynb`:

* `fastai2`  installed directly into the instance with:
  *  `!pip install fastai2`
* `fastai2_audio` installed directly into the instance with:
  * `!pip install git+https://github.com/rbracco/fastai2_audio.git` 
* In addition a Librosa Soundfile dependency must be installed:
  * `!conda install -c conda-forge libsndfile --yes`

## Dataset

The procedure to download the Freesound 2019 Kaggle Competition dataset directly to the notebook instance storage is outlined within [this](https://www.kaggle.com/general/74235) Kaggle forum post and followed in detail within the notebooks stated above.



A huge thanks to the fastai & fastai audio communities and those who produced Kaggle Competition write-ups in being so generous and  informative with their efforts, knowledge, and time.

Should you have any queries regarding this repo, please contact me via a GitHub Message.