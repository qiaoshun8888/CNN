### Using TensorFlow or Keras 
TensorFlow: https://github.com/fomorians/distracted-drivers-tf  (1.04)
Keras: https://github.com/fomorians/distracted-drivers-keras  (1.26)
Discussion Link: https://www.kaggle.com/c/state-farm-distracted-driver-detection/forums/t/20129/cloud-gpu-starter-project

### Using nolearn/lasagne
Repo: https://github.com/ottogroup/statefarm.
Discussion Link: https://www.kaggle.com/c/state-farm-distracted-driver-detection/forums/t/20482/getting-started-with-nolearn-lasagne
- Did right cropping the images help ?
It was mostly to get square images and the right side seemed to contain the most relevant information. But maybe the net could also learn the correct label from looking at the notepad on the woman's leg :)


### Simple solution (Keras) by ZFTurbo
Latest version: https://github.com/ZFTurbo/KAGGLE_DISTRACTED_DRIVER/blob/master/run_keras_cv_drivers_v2.py
1. Code randomly rotate images +-10 degrees
2. Code uses the same CNN structure from mentioned post with Dropout layers after each Conv/Pool layer. This allows to slightly reduce overfit.
3. Code uses 64x64 pixel grayscale images
4. CNN is actually simple enough to be run on ordinary computer in reasonable time
5. I added some useful callback functions: EarlyStopping - to stop early after loss stop decreasing ModelCheckpoint - save best weights and restore them for minimum loss after "fit" ends. Some kind of XGBoost's best_ntree_limit

Notes:
- Crossfold score is actually much lower than leaderboard one
- In most cases best loss achieved right after first epoch
- I feel like this line:

`train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)`

should be replaced with some other function. It would be good if someone propose best replacement.

- "batch_size" quite strongly affects learning process
- It seems KFold = 26 is best for this problem.

Running "as is" from repository will generate submission with validation loss around 0.27 and LB score ~1.03. I was able to generate solutions with 0.85-0.9 score with same code on different parameters. But I wasn't experimented much.
Discussion Link: https://www.kaggle.com/c/state-farm-distracted-driver-detection/forums/t/19971/simple-solution-keras

