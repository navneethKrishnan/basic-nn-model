# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
The Neural network model contains input layer,two hidden layers and output layer.Input layer contains a single neuron.Output layer also contains single neuron.First hidden layer contains six neurons and second hidden layer contains seven neurons.A neuron in input layer is connected with every neurons in a first hidden layer.Similarly,each neurons in first hidden layer is connected with all neurons in second hidden layer.All neurons in second hidden layer is connected with output layered neuron.Relu activation function is used here .It is linear neural network model(single input neuron forms single output neuron).

## Neural Network Model


![nn](https://user-images.githubusercontent.com/75235477/187118781-23269b91-2f69-44d2-85bd-78e0a41757b1.png)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```python
# Developed By:Tamil Venthan R S
# Register Number:212220230054
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df=pd.read_csv("data2.csv")
df.head()
x=df[['input']].values
x
y=df[['output']].values
y
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=40)

scaler=MinMaxScaler()
scaler.fit(xtrain)
scaler.fit(xtest)
xtrain1=scaler.transform(xtrain)
xtest1=scaler.transform(xtest)

model=Sequential([
    Dense(6,activation='relu'),
    Dense(7,activation='relu'),
    Dense(1)
])
model.compile(optimizer='rmsprop',loss='mse')
model.fit(xtrain1,ytrain,epochs=4000)
lossmodel=pd.DataFrame(model.history.history)
lossmodel.plot()
model.evaluate(xtest1,ytest)

xn1=[[56]]
xn11=scaler.transform(xn1)
model.predict(xn11)
```

## Dataset Information
![2022-08-28](https://user-images.githubusercontent.com/75235477/187083773-f50a5abe-2abc-4204-b5c1-1d0cd1e9775c.png)


## OUTPUT

### Training Loss Vs Iteration Plot
![2022-08-28 (3)](https://user-images.githubusercontent.com/75235477/187083896-b57d1a9e-e9f5-4204-9160-21bb24c81d9e.png)



### Test Data Root Mean Squared Error

![1dl](https://user-images.githubusercontent.com/75235477/187083843-90d823ba-148e-49a8-a65a-57dd3b11f4b4.png)


### New Sample Data Prediction

![2022-08-28 (2)](https://user-images.githubusercontent.com/75235477/187083819-6a184b26-faec-4266-a8ef-ca188167e10e.png)


## RESULT
Thus,the neural network regression model for the given dataset is developed.
