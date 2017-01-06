# Steering a car

Here is behavioral cloning project, that I did as part of [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/drive).
Behavioral cloning is technics for teaching neural networks to do useful things or behave in desired way.
In this example we need to steer a car while driving in simulator 
([Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip),
[MacOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip),
[Win32](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip),
[Win64](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip))
based onboard camera images. So firstly we need to drive the car ourselves and collect the images from the camera and
corresponding steering wheel angle positions. Then we need to teach neural network to return steering angle for provided
image - well known and tested supervised learning task. Sounds pretty simple.

# How does it work?

There is nice [Nvidia paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
where they applied this approach to drive a real car. You may find videos of it on [youtube](https://www.youtube.com/results?search_query=nvidia+bb8). It's pretty amazing to watch.
I use the same but simplified method. Since we are working with images we need
[convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) or CNN which is specialized on image processing.
There are a lot of free tutorials over the internet about CNNs. [For example](https://www.udacity.com/course/deep-learning--ud730).
So I decided to use similar to Nvidia CNN.

![CNN architecture](https://github.com/parilo/steering-a-car-behavioral-cloning/blob/master/Proj3-CNN.png "CNN architecture")

Why this architecture? So, I used grayscaled and normalized to interval [-0.5, 0.5] images to reduce memory and GPU usage on training. That's why we have 1 channel image as input.
I tried to use minimalistic neural network but it's hard to say that it is tiny network. It has 4 151 384 parameters. 182 328 and 3 969 056 parameters
for CNN and fully connected part respectively. It is big number of parameters and additional experimenting probably may decrease that number. Also I
used fully connected layer with 5 hidden layers to allow fully connected network to have its own lower, middle and high level features as it is
done in CNN. Also I used [dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout) with 0.5 probability on all layers except last 3 
relatively small layers to deal with [overfitting](https://en.wikipedia.org/wiki/Overfitting).

# Training data

For all supervised learning tasks such as our behavioral cloning collecting training data has an extremely high importance. We need to collect dataset which
is correctly represent all possible situations that can emerge while driving. I mean situations where we not only driving a car straight across the street and
making turns left or right. But also situations where we need to recover a car from bad positions on the road. Such as various course deviations. As we mostly
drive a car straight such samples of straight driving will probably dominate the dataset and lead to model overfitting. I decided to record number of datasets with
different behavior and mix them togeather:
- Straight driving with turns
- Strong disturbance recovery
- Medium disturbance recovery
- Light disturbance recovery
I recorded disturbance recovery dataset by randomly making disturbance to right and recovery with left steering. So then I just excluded all right turns from that dataset.
Also I repeated that procedure with left turns. So at the end I had 6 dataset with disturbance recovery. 3 for recovery from right and 3 for recovery from left.
I mixed a 30% strong, 100% of medium and light disturbance and excluded 90% of straight drive samples in result dataset. As a result I have 11841 samples in my dataset.
There is 2 tracks in simulator. I used only track 1 for recording samples. Track 2 will be used to test the model how good it generalized steering a car.

![Steering distribution in the dataset](https://github.com/parilo/steering-a-car-behavioral-cloning/blob/master/dataset-steering-distribution.png "Steering distribution in the dataset")

Left picture shows steering distribution in the dataset and right picture shows steering distribution in augmented dataset (read further for detalis)

# Dataset augmentation

11841 samples is not big enough dataset to train a good quality model because of overfitting. That's why we need to augment our dataset with generated samples. Our sample is image and corresponding steering wheel.
Here is an example of input image.

![Dataset image example](https://github.com/parilo/steering-a-car-behavioral-cloning/blob/master/dataset-image-example.png "Dataset image example")

So we need to generate samples with images and corresponding steering wheel positions. I decided to use these transformations that leaves steering wheel untouched:
- randomize image brightness (-0.3, 0.3)
- randomly partially occlusion with 30 black 25x25 px squares
- very slightely randomly:
    - rotation 1 degree amplitude
    - shift 2px amplitude
    - scale 0.02 amplitude
And
- flipping the image with corresponding flipping of steering wheel
Also I disturbed steering wheel value with small normal noise (0 mean, 0.005 standard deviation). You can see resulting steering wheel distribution on the right dataset steering distribution image earlier.
Augmented images looks like this:

![Augmented image example](https://github.com/parilo/steering-a-car-behavioral-cloning/blob/master/augmented-image-example.png "Augmented image example")

# Training

For creating and training the model I used [Keras](https://keras.io/) which has big library of standard neural networks blocks. So training neural networks with Keras is pretty simple and fun :)
For training I used only augmented samples, so model haven't seen one sample twice. That is again for preventing the overfitting. I used Adam optimizer with 1e-4 learning rate and mean squared error as loss function.
I decided to use 96 samples batch size and 25 epochs of 38400 samples. Collected dataset I splitted into 67% train and 33% validation parts. As test dataset I used straight driving samples recorded on track 2.
I saved model on every epoch and selected one from last epochs models that is able to drive track 2. I tried several times to train and noticed that not every time it is possible to select such model. But models are close to drive track 2.
Despite of track 2 haven't been used to record samples. And has much sharper turns (and higher complexity as for me). Training model can drive it without seeing a single image from it. That fact was very surprising for me.

# How to train and run?

- Download simulator ([Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip),
[MacOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip),
[Win32](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip),
[Win64](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip))

- Record your dataset. It is csv file with such records
    center image path, left image path, right image path, steering, throttle, breakvals, speed
    only center image path and steering column is required

- Use model.py or Behavioural-Cloning-clean.ipynb ipython notebook to train your model.
    dont forget to change dataset loading part to specify your dataset location.
    GPU is strongly recommended.

- Select autonomous mode and track in simulator. Run command
```
python drive.py model.json
```
Already trained model is included. You can try it :)

# Conclusion

I found this approach very useful and I think it can be used in other simulators and games such as GTA or TORCS. Or even other type of games maybe.
Also I think it can be used as pretraining of actor in actor-critic reinforcement learning and may significantly decrease
learning in RL.
