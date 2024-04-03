# multispectral

The "brain_tumor" repository contains the data and analyses of hyperspectral brain scans. The data is publicly available at https://hsibraindatabase.iuma.ulpgc.es/ and is described in https://ieeexplore.ieee.org/document/8667294. 

Data reduction is done in "reduce_HSI.ipynb"
- The data preprocessing procedure generally follows that done by Fabelo et al. (2019). The original raw data consists of 826 frequency bands separated by ~0.7 nm. The first and last few frequency bands were omitted to avoid large instrumental uncertainties. The spectral dimension was further reduced to a sampling of ~7 nm by averaging contiguous bands into 65 coarser bands. white and dark images were used for flat fielding and zero-point corrections, respectively. Finally, each pixel was normalized to have a maximum value of unity and a minimum of zero.

- The overall goal is to train a model with supervised learning to classify a type of tissue given a hyperspectral image cube. The dimensions of each cube was 17x17x65 (17 pixels spatial; 65 spectral) centered on the pixel of interest.
