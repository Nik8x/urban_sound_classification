# Urban Sound Classification

**Live report:** https://nik8x.github.io/urban_sound_classification/

Classifying 10 categories of urban sound (air conditioner, car horn,
children playing, dog bark, drilling, engine idling, gun shot,
jackhammer, siren, street music) from
[UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html).

This works from a 450-clip stratified sample (45 per class), fetched
directly through the HuggingFace `danavery/urbansound8K` mirror's
per-row signed audio URLs via the datasets-server API, so nothing close
to the full ~6.5GB archive needs to sit on disk. Notebook 00 fetches
this automatically if `data/manifest.json` doesn't exist yet.

## Notebooks

1. `00_data_setup_eda.ipynb`: fetches the stratified sample, waveform
   and mel-spectrogram examples per class, duration and sample-rate
   audit.
2. `01_statistical_testing.ipynb`: duration and RMS loudness both
   differ significantly by class (Kruskal-Wallis, both p &lt; 0.0001).
3. `02_feature_engineering_selection.ipynb`: a 120-dim MFCC feature
   vector (mean, standard deviation, and delta of 40 coefficients),
   plus fixed-size mel-spectrograms for a CNN.
4. `03_model_training_evaluation.ipynb`: random forest on the MFCC
   features and a Conv2D/MaxPooling2D CNN on the spectrograms, both
   evaluated on the same stratified held-out test split.
5. `04_clustering.ipynb`: KMeans and Gaussian mixture clustering by
   MFCC profile alone, checked against the true classes afterward.

## Results

Random forest on the fixed MFCC features reaches 74.4% accuracy (macro-F1
0.743) against a 10% majority-class baseline. The CNN on spectrograms
only reaches 41.1%, a real, honest result given only 36 training clips
per class, not evidence that CNNs are worse for audio in general, at
UrbanSound8K's full scale the balance would likely tip the other way.
Unsupervised clustering only weakly recovers the 10 classes (ARI
0.10-0.11), a clear contrast to how well the supervised model does with
labels to learn from.

Full write-up with charts: `docs/index.html` (also published via GitHub
Pages).

## Future work

- Scale up to the full UrbanSound8K corpus on a machine with enough disk
  to hold it (about 6.5GB), to see whether the CNN overtakes the
  classical model at full scale, as expected.
- Try transfer learning from an audio-pretrained model (VGGish, YAMNet)
  instead of training a CNN from scratch on so little data.
- Use UrbanSound8K's official 10-fold cross-validation split (grouped by
  source recording) instead of a plain random split.
