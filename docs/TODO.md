# Workflow
## reading
- [x] read-up on current analysis (what is done, how is it performed)

## data preprocessing
- [ ] create data and metadata model
- [ ] import all unlabeled data
- [ ] create data overview
- [~] match labeled data

- get label points for detecting somas;
- https://github.com/cvat-ai/cvat

## labeling
- [ ] semi-automatic labeling of neurons; Create labeled dataset;
  - LabelStudio (https://labelstud.io/guide/labeling); Auto-Annotation with machine learning based model;
  - https://www.cvat.ai/ CVAT
  - MONAI label  https://project-monai.github.io/core.html#getting-started


## image processing pipeline
- [ ] implement first software pipeline based on transformer models
- [ ] evaluation on test and trainings data

## Sholl analysis
- [ ] calculate metrics on segmented neurons and dentrites
- [ ] calculate metrics on images

## reporting
- create interactive report (webpage for visualization)

## documentation
- [ ] create PDF report about project
- [ ] create presentation


# Results
## Literature review
### What is Sholl analysis
Sholl analysis is a simple, interpretable feature-extraction method rather than a deep-learning method. A good modern workflow would be: segment/traces neurons or neurites with deep learning, then compute Sholl-derived features as interpretable morphology endpoints.

https://imagej.net/imagej-wiki-static/Sholl

*Original method*
Sholl DA. 1953. “Dendritic organization in the neurons of the visual and motor cortices of the cat.” Journal of Anatomy. Classic origin of the concentric-shell/circle approach.

*Conceptual modern explanation*
Bird & Cuntz. 2019. “Dissecting Sholl Analysis into Its Functional Components.” Cell Reports. Very useful for understanding what the Sholl profile actually encodes: arbor domain, total dendrite length, and angular/root-angle structure.

*Practical / advanced variants
O’Neill et al. 2015. “Assessing effects on dendritic arborization using novel Sholl analyses.” Frontiers in Cellular Neuroscience. Good for understanding limitations of standard Sholl analysis and why local/sub-arbor analyses may be needed.

*Automatic workflows*
Langhammer et al. 2010. “Automated Sholl analysis of digitized neuronal morphology at multiple scales.” Cytometry A. Describes Bonfire and whole-cell vs. subregion analysis.

*ImageJ/Fiji implementation*
Ferreira et al. 2014. “Neuronal morphometry directly from bitmap images.” Nature Methods. This is listed by the ImageJ Sholl documentation as the authoritative reference for the Fiji Sholl Analysis implementation.

*Software comparison*
Binley et al. 2014. “Sholl analysis: a quantitative comparison of semi-automated methods.” Useful because it emphasizes that automated methods may differ and require manual calibration/validation.


![./images/Scholl1953_Fig8.png]

### Example analyses (SDU)
Two examples for analysis are Mortensen2024 and Simonsen2025. The general analysis is very similar using a single circle around the neuron bodies for analysis.

Mortensen et al. 2024, Neuropharmacology — “Modeling mechanisms of chemotherapy-induced peripheral neuropathy and chemotherapy transport using induced pluripotent stem cell-derived sensory neurons.” This SDU-affiliated study used Sholl analysis in ImageJ to quantify the number of axons emanating from iPSC-derived sensory-neuron ganglia after paclitaxel/vincristine exposure.

> We assessed neurotoxicity by measuring the area of the neuronal network and the number of axons emanating from each ganglion using Sholl analysis (ImageJ software 2.0.0). The analysis was blinded to the person performing the Sholl analysis to reduce risk of bias. We obtained three images per triplicate conditions from 1 to 2 independent differentiations. All images were converted to 8-bit, and an appropriate threshold specific for each differentiation was applied. The center of the ganglion was defined using the straight-line tool, and the Sholl analysis was performed with an appropriate end-radius for all ganglia in each image (Supplementary Fig. S1). The end-radius was adjusted manually according to the size of the ganglia. The number of ganglia with >30 axons was subsequently counted for each chemotherapy concentration and is presented relative to the total number of ganglia counted for each chemotherapy concentration.

![./images/Mortensen2024_Fig3.png]
![./images/Mortensen2024_Fig5.png]
![./images/Mortensen2024_Fig6.png]

Simonsen et al. 2025, Basic & Clinical Pharmacology & Toxicology — “The Molecular and Clinical Impact of Atorvastatin Exposure on Paclitaxel Neurotoxicity in Sensory Neurons and Cancer Patients.” This translational SDU-associated paper quantified axons emanating from each ganglion using Sholl analysis in iPSC-derived sensory neurons.


These images contain dose-dependent exposure of neurons to chemotherapy. We are looking for structural changes in the network, could be blabbing, reduction of neuronal network, number of neurons etc. We used to do Sholl analysis, because we find good dose-dependent effects, especially for paclitaxel, vincristine, and bortezomib, but we do it manually and it is a nightmare (for the students). Be happy to see what you can make of it.


### AI based tools
#### NeuroQuantify — deep learning for cells and neurites

NeuroQuantify is directly relevant for your question. It is an open-source deep-learning tool for detecting neuron cells and neurites in microscopy images, especially phase-contrast neuronal network images. It uses a modified U-Net-style model for cell/neurite segmentation and then computes quantitative neurite features such as neurite length and orientation. It is not primarily a Sholl package, but its output is the kind of segmentation that could be used upstream of Sholl analysis.

Best use case: 2D in vitro neuronal cultures, especially when you need automated soma/cell and neurite segmentation.

Limitation: it focuses on cells/neurites and neurite length/orientation, not necessarily full dendritic tree reconstruction per individual neuron in dense overlapping cultures.

- https://github.com/StanleyZ0528/neural-image-segmentation; good python code available; 
- 3 years no contributions

#### SENPAI — multiscale neuronal segmentation, including spines

SENPAI, published in 2024 in Nature Communications, is a modular framework for multi-scale tissue imaging and analysis. The authors report segmentation from whole neurons down to spines and state that it outperforms state-of-the-art tools for these tasks. This is closer to a modern AI-based end-to-end neuronal morphology workflow, especially for complex neuronal images.

Best use case: high-resolution neuronal tissue imaging, where you need robust segmentation across scales.

Limitation: depending on your image type, it may require adaptation/training and is not simply a drop-in “Sholl button.”

- pretty good, but MATLAB tool
