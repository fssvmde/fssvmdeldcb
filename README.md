# Diffusion-Based Any-to-Any Voice Conversion 

Pytorch implementation of the paper ``Diffusion-Based Voice Conversion with Fast Maximum Likelihood Sampling Scheme''.

# Voice conversion with the pre-trained models

Please check *inference_multi.ipynb* for the instructions.

# Training your own model

Brief instructions:
1. If you want to train the encoder, please run *train_enc.py*. Before that, you have to prepare two folders: "mels" with the input mel-spectrograms and "mels_avg" -- with those of the ``average voice''. To obtain the latter, you have to run Montreal Forced Aligner (or any other alignment algorithm of your choice) on the input mels. Please check our paper for more details on how the ``average voice'' is built.
2. Run *train_dec.py* to train the decoder. Before that, you have to prepare three folders: "mels" with mel-spectrograms, "wavs" with raw audio files and "embeds" with 256-dimensional speaker embeddings extracted by the pre-trained speaker verification network.

More detailed instructions will be provided soon.
