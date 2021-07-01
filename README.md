# Facemask-Detector

Made this face mask detector using Convolutional Neural Networks with the help of haarcascade face recogonitions models.

<h3><b>Improved the accuracy from 96% to 98.7%</b></h3>

### Dataset
[Link](https://drive.google.com/file/d/1MRTO03WAvc69un-93UhCFErsW3ZtZp6T/view)
This is an open dataset that contains approx 12k images divided into different direcotries for train and testing the model. <br />
Inside each directory, there are separate directories for masked images and un-masked images.

### About the Model
It consists of 3 Conv2D layers, each followed by a MaxPooling Layer, followed by a Flatten and Dense layer, and finally the Output Layer. <br />
<img src="https://github.com/Kushagraw12/Facemask-Detector/blob/main/Helper%20Images/model.png" center/><br />
I have used ```'relu'``` activation layer and ```sigmoid``` activation for the output layer. <br />
I have used the input size as ```150 x 150 x 3```.<br />
<!-- <img src="https://github.com/Kushagraw12/Facemask-Detector/blob/main/Helper%20Images/model2.jpg" /><br /> -->
All of these configurations are from either hit and trial method or from my past experiences. (suggestions are welcome!)

### Loss Function
The model usses ```binary_crossentropy``` loss function. <br />
I was able to achieve good accuracy and low loass using this loss function, and the final outcome was great! <br /> 
<img src="https://github.com/Kushagraw12/Facemask-Detector/blob/main/Helper%20Images/graph%20loss.png" />
<img src="https://github.com/Kushagraw12/Facemask-Detector/blob/main/Helper%20Images/graph%20accuracy.png" />

### Partner
I worked on this project with my friend <br />
Himanshu Sheokand [ [LinkedIn](https://www.linkedin.com/in/himanshu-sheokand-595786182/) ] | [ [GitHub](https://github.com/himanshu2030) ] <br />

```Happy coding!```
