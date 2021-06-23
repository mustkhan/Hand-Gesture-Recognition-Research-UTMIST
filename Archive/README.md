# Hand-Gesture-Project
## How to use saved weights
In test-app.py
1. Line 26: Make sure *classes* contains the correct labels
2. Line 72: Make sure the output size is correct
```javascript
self.out = tf.keras.layers.Dense(7, activation='softmax', name="output")
```
3. Line 96: Make sure the correct weight is used
```javascript
new_model.load_weights('weights/path_to_my_weights_class7complex3')
```
4. **Run** python test-app.py

#### Result (4 Classes)
![](/Image/ezgif-3-2b7e533314b2.gif)

## How to train a model
Make sure folder *training_sample* and *validation-sample* is empty
#### Collect Training data
In training-sample.py 
1. Line 24, 25: Change path
```javascript
[//]: The folder where we will put all our training samples
path_dest = 'C:/Users/chaoj/Desktop/UTMIST/3D-CNN-Gesture-recognition/training_samples/' 
[//]: The folder that contains all the data (Jester 20bn)
path_source = 'C:/Users/chaoj/Desktop/UTMIST/Jester/20bn-jester-v1/' 
```
2. Line 32: Make sure **class_** contain the right labels
3. **Run** python training-sample.py

#### Collect Validation data
In validation-sample.py
1. Line 23, 24: Change path
```javascript
[//]: The folder where we will put all our validation samples
path_dest = 'C:/Users/chaoj/Desktop/UTMIST/3D-CNN-Gesture-recognition/validation_samples/' 
[//]: The folder that contains all the data (Jester 20bn)
path_source = 'C:/Users/chaoj/Desktop/UTMIST/Jester/20bn-jester-v1/'
```
2. Line 32: Make sure **class_** contain the right labels
3. **Run** python validation-sample.py

#### Train Complex model in main.ipynb

## NOTE
If you want to train a model with different number of classes, remember to 
1. Update the output of the model to the appropriate number
```javascript
self.out = tf.keras.layers.Dense(7, activation='softmax', name="output")
```
2. Delete folder *training_today*, *training_checkpoints*, *.idea*, and *.ipynb_checkpoints*
3. Run through the simple model (epoch = 1) once to create checkpoints
3. Run the complex model
4. Save weights
5. Test weights

## Results
~ 85% Accuracy with 4 classes (epoch = 3) 
weights: 
```javascript
new_model.load_weights('weights/path_to_my_weights_complex3')
```
![GitHub Logo](/Image/class4epoch3.png)

~ 55% Accuracy with 7 classes (epoch = 3)
weights: 
```javascript
new_model.load_weights('weights/path_to_my_weights_class7complex3')
```
![GitHub Logo](/Image/class7epoch3.png)

## Next Step :rocket:
1. Increase Accuracy
    * Increase epoch
    * Modify model (maybe)
    
2. Think of ways to visualize the model


## Reference:
[3D-CNN](https://github.com/anasmorahhib/3D-CNN-Gesture-recognition)
