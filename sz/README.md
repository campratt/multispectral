# README
This project uses simulated multispectral patches of the microwave sky as seen by the Planck and WMAP satellites to extract the thermal Sunyaev-Zel'dovich (SZ) effect which is a distortion of the cosmic microwave background (CMB). 

Data reduction:
- The data generation and preprocessing are not included, however, it will be provided in an upcoming publication. The general idea is to create 100 simulated datasets of the microwave sky with different components of emission: thermal dust, synchrotron, infrared sources, radio sources, CMB, and the SZ effect. Instrumental uncertainty was included as well. The different components were convolved with the spectral and beam responses of the Planck and WMAP telescopes. These data were generated using the PySM3 software as well as the data produced by Han et al. (2021). Each full-sky realization consisted of 12 mock frequency observations, ranging from 23-353 GHz. These were then broken into 250 randomly selected square patches over the sky; each patch was 128x128 pixels with a resolution of 2 arcminutes per pixel and this creates the "data/X" training data. The "data/Y" labeled data were the same patches of sky but for the SZ effect. In total, we used 80 full-sky realizations for training and 20 for testing. The full dataset is not included, however, a single example of the test data is provided.

- The main idea was to use simulated data to train a model that extracts the SZ signal from a patch of frequency data, and this is done in "code/training.py". The model architecture included 3D convolutional layers followed by a pointwise convolution to produce a single channel 2D map of the SZ effect. The mean squared error was minimized with the Adam optimizer, and the final model was selected at the epoch with the smallest validation loss.

- Plotted in "evaluate_model.ipynb" and "model_performance.png" shows the model prediction, ground truth, and the residuals between them.

## Caveats about this project:
1. 

