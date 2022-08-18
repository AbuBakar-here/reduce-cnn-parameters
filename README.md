# Reducing Training Parameters in CNNs for Efficient Learning

I reduced the number of parameters used in a deep learning architecture while maintaining same accuracy.

## Motivation

I am a noob in deeplearning, all I knew was **convolutional neural networks** are very effective in understanding an image but the models I've made have a lot of **training parameters**. So I asked myself, how about I create a model with fewer parameters while maintaining the same accuracy as before? 

## Implementation

For the sake of simplicity, I only did two things for pre-processing:
1. Used `tensorflow.keras.utils.to_categorical` on labels.
2. Divided pixels values by `255` for **normalization**.

<br />

Also there will be three models for benchmark:
1. Baseline
2. Approach 1
3. Approach 2


## Results

Few things to keep in mind before we jump to results:

* I have choosed [MNIST dataset](https://www.kaggle.com/c/digit-recognizer) for this experiment.
* The baseline model I choosed have ~570,000 parameters, normally others have around 100,000-200,000 parameters.
* Earlystop with parameters `monitor='val_loss', patience=2` is also used.
* No padding is used as the information in all images are present in centre.

Model | Parameters | Layer count | Training Accuracy
--- | --- | --- | ---
**Baseline** | 591786 | 5 | 98%
**Approach 1** | 3418 | 10 | 98%
**Approach 2** | 3130 | 9 | 97%

It seems *Approach 1* was successful. I was able to maintain same accuracy and reduce the parameters to 3418 which is **less than 1% of parameters than baseline model**.

## What I learned?

* `MaxPool2D` layer helps in parameter reduction.
* More layers does not neccessarily means good results.
* Regulurization i.e. `BatchNormalization`, `Dropout`, etc is good but too much can lead to underfitting.

## Use

Click below to open in Google Colab or clone the repository.

[![Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/AbuBakar-here/reduce-cnn-parameters/blob/main/CNN_on_MNIST.ipynb)

## License

> You can check out the full license [here](https://github.com/AbuBakar-here/reduce-cnn-parameters/blob/main/LICENSE)

This project is licensed under the terms of the **MIT** license.

## Reference

[How to reduce training parameters in CNNs while keeping accuracy >99%](https://towardsdatascience.com/how-to-reduce-training-parameters-in-cnns-while-keeping-accuracy-99-a213034a9777)

> I came across this article while researching which better describes the reduction of CNNs parameters while preserving accuracy.
