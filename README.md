# behavorial_cloning

The architecture adopted for solving this problem was similar to the solution presented by Bojarski, Mariusz, et al [1].
It consists of 5 Convolutional Neural Networks followed by 3 Dense Neural Networks as follows:

Input size: 80 , 160 , 3
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param n.o     Connected to                     
====================================================================================================
convolution2d_47 (Convolution2D) (None, 38, 78, 24)    1824        convolution2d_input_16[0][0]     
____________________________________________________________________________________________________
convolution2d_48 (Convolution2D) (None, 17, 37, 36)    21636       convolution2d_47[0][0]           
____________________________________________________________________________________________________
convolution2d_49 (Convolution2D) (None, 7, 17, 48)     43248       convolution2d_48[0][0]           
____________________________________________________________________________________________________
convolution2d_50 (Convolution2D) (None, 5, 15, 64)     27712       convolution2d_49[0][0]           
____________________________________________________________________________________________________
convolution2d_51 (Convolution2D) (None, 3, 13, 64)     36928       convolution2d_50[0][0]           
____________________________________________________________________________________________________
flatten_10 (Flatten)             (None, 2496)          0           convolution2d_51[0][0]           
____________________________________________________________________________________________________
dropout_28 (Dropout)             (None, 2496)          0           flatten_10[0][0]                 
____________________________________________________________________________________________________
fc1 (Dense)                      (None, 100)           249700      dropout_28[0][0]                 
____________________________________________________________________________________________________
dropout_29 (Dropout)             (None, 100)           0           fc1[0][0]                        
____________________________________________________________________________________________________
fc2 (Dense)                      (None, 50)            5050        dropout_29[0][0]                 
____________________________________________________________________________________________________
dropout_30 (Dropout)             (None, 50)            0           fc2[0][0]                        
____________________________________________________________________________________________________
fc3 (Dense)                      (None, 10)            510         dropout_30[0][0]                 
____________________________________________________________________________________________________
output (Dense)                   (None, 1)             11          fc3[0][0]                        
====================================================================================================
Total params: 386619


Dropout layers were included to prevent over-fitting. Relu units were used throughout the neural network architecture in order to make the system compatible with non-linearities and thus able to represent more complex systems.

The original dataset provided by Udacity was used to train the neural network.
When running an initial experiment using all the dataset, the trained model was driving the car off the road. 
After a careful and detailed analysis of the dataset, it was realised that there were too many samples centred around steering = 0 and too few at extreme regions (-1.0 and 1).
The key method to make the car stay on the track was to remove the excess of training data in the centre region and add copies of data at extreme regions.
The images were also reduced to half the size in order to make the training steps faster and normalised between -0.5 and 0.5.
From the original 8036 images (from centre camera), only 398 were used for training and 100 for validation.

In the end 50 epochs were enough to train (under one minute!) a model which is able to stay on track.

This architecture proved to be adequate for the given problem and has the main advantages of being simple, small and easy to train in very few steps and with a very small dataset.

Note: I decided not to augment the dataset by shifting or rotating existing images because that introduces unpredictable and artificial biases which may lead to misbehaviours as the paper authors also mention.

References:
[1] “Bojarski, Mariusz, et al. "End to End Learning for Self-Driving Cars." arXiv preprint arXiv:1604.07316 (2016).
