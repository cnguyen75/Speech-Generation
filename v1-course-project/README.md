# v1 (course project)

This folder keeps the original version of the project, written for Math 132A (Professor Paul J. Atzberger) and dated March 18, 2025. The original README is unchanged in [`ORIGINAL_README.md`](ORIGINAL_README.md), and the notebook and images are as they were uploaded. The study in the repository root (v2) replaces it.

## Limitations of v1

Read the original README with these in mind:

- **The model is not Tacotron 2.** `speech_generation.ipynb` trains `Tacotron2Placeholder`, a two-layer MLP (Linear 400→1024, ReLU, Linear→80). WaveGlow is mentioned in the README but does not appear in the code.
- **The input is not the text.** Each clip's "text embedding" is `torch.randn(400)`, a random vector, so the model cannot learn to say anything.
- **Validation overlaps training.** Training uses the first 100 clips of LJSpeech (`lines[:100]`), and validation uses the first 20 (`lines[:20]`) of the same file. Validation loss therefore measures fit to training clips.
- **The README's results table does not match the notebook.** The table reports validation loss and MCD of 4.60 / 2047.76 (Adam), 9.06 / 2303.04 (L-BFGS) and 16.96 / 2303.04 (SGD). None of these values appear in the notebook outputs. The 100-epoch run in cell 6 ends at:
  - Adam: validation loss 4.8618, "MCD" 2045.8933
  - SGD: validation loss 11.3516, "MCD" 2255.9436
  - L-BFGS: validation loss 130.6736, "MCD" 2125.7812. Its training loss was 0.7771, so it overfit badly.

  A separate 10-epoch run with all five optimizers (cell 9) ends with validation losses between 0.2750 and 0.4061 for the first-order methods, and 20.5259 for L-BFGS, which never moved. The first SGD attempt (cell 3) diverged to `nan` after one epoch.
- **The "MCD" is not a mel-cepstral distortion.** `compute_mcd` takes logs of the mel-spectrogram values and sums squared differences over frames and bands. It never computes cepstra or aligns frames. That is why the values are around 1,800–2,500, while the README's own reference point for human-like speech is about 5.
- **Training was CPU-only and short.** Generated speech was unintelligible, as the original README says.

The idea, comparing SGD, Adam, AdamW, RMSprop and L-BFGS on a speech-synthesis loss, carries into v2. v2 uses a real pretrained Tacotron 2, real text, separate train / validation / test splits, tuning on validation only, five seeds, and audio you can listen to.
