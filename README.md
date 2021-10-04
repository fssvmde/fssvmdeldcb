# Diffusion-Based Any-to-Any Voice Conversion 

Pytorch implementation of the paper "Diffusion-Based Voice Conversion with Fast Maximum Likelihood Sampling Scheme".

# Voice conversion with the pre-trained models

Please check *inference_multi.ipynb* for the instructions.

The pre-trained universal HiFi-GAN vocoder we use is available at https://drive.google.com/file/d/10khlrM645pTbQ4rc2aNEYPba8RFDBkW-/view?usp=sharing. Please put it to *checkpoints/vocoder/universal/*

You have to download voice conversion model trained on LibriTTS from here: https://drive.google.com/file/d/18Xbme0CTVo58p2vOHoTQm8PBGW7oEjAy/view?usp=sharing

Additionally, we provide voice conversion model trained on VCTK: https://drive.google.com/file/d/12s9RPmwp9suleMkBCVetD8pub7wsDAy4/view?usp=sharing

Please put voice conversion models to *checkpoitns/vc/*

# Training your own model

Brief instructions:

1. If you want to train the encoder, please run *train_enc.py*. Before that, you have to prepare two folders: "mels" with the input mel-spectrograms and "mels_avg" - with those of the "average voice". To obtain the latter, you have to run Montreal Forced Aligner (or any other alignment algorithm of your choice) on the input mels. Please check our paper for more details on how the "average voice" is built.

2. Run *train_dec_multi.py* to train the decoder. Before that, you have to prepare three folders: "mels" with mel-spectrograms, "wavs" with raw audio files and "embeds" with 256-dimensional speaker embeddings extracted by the pre-trained speaker verification network located at *checkpoints/spk_encoder*

The functions for extracting speaker embedding and calculating mel-spectrograms can be found at *inference_multi.ipynb* (*get_embed* and *get_mel* correspondingly). More detailed instructions will be provided soon.
