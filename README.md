# multispectral

The "brain_tumor" repository contains the data and analyses of hyperspectral brain scans. The data is publicly available at https://hsibraindatabase.iuma.ulpgc.es/ and is described in https://ieeexplore.ieee.org/document/8667294. 

Data reduction is done in "reduce_HSI.ipynb"
- The data preprocessing procedure generally follows that done by Fabelo et al. (2019). The original raw data consists of 826 frequency bands separated by ~0.7 nm. The first and last few frequency bands were omitted to avoid large instrumental uncertainties. The spectral dimension was further reduced to a sampling of ~7 nm by averaging contiguous bands into 63 broad bands. white and dark images were used for flat fielding and zero-point corrections, respectively. Finally, each pixel was normalized to have a maximum value of unity and a minimum of zero.

- The overall goal was to train a model to classify tumor or normal tissues given a hyperspectral image cube with supervised learning. The dimensions of each cube was 17x17x63 (17 pixels spatial; 63 spectral) centered on the pixel of interest. The original labeled data contains 4 categories: normal tissue, tumor tissue, hypervascular tissue, and background objects. Hypervascularized tissue is easily distinguishable by eye and background material is trivial. Separating tumor from normal tissue, however, is less obvious and is much more valuable information to a surgeon. To this end, we focus only on tumor and normal tissues for classification. The number of tissue examples were evenly sampled for each patient. Training and testing samples consisted of the first 21 and last 8 patients respectively.

- A 3D convolutional neural network
