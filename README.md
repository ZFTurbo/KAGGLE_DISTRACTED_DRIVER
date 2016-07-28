# KAGGLE_DISTRACTED_DRIVER

Version 1: run_keras_simple.py - simple solution using Keras

Version 2: run_keras_cv.py - Added cross-validation (LB ~1.4)

Version 3: run_keras_cv_drivers.py - Added cross-validation based on driver ID

Version 4: run_keras_cv_drivers_v2.py - Added random rotation of images, new CNN structure and some useful callbacks like "EarlyStopping" and "ModelCheckpoint". (LB: ~1.0 and lower)

Version 5: kaggle_distracted_drivers_vgg16.py - Using pretrained VGG16 CNN. Allows to get 0.203 on LB.
Requirements: 16 GB of RAM with SWAP enabled, powerful NVIDIA GPU, Keras 0.2.0 (it's important).
Weights: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
