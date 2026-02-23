# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="832" height="834" alt="image" src="https://github.com/user-attachments/assets/eb33c3b3-69b8-437a-82b4-e892642f907b" />

## DESIGN STEPS

### STEP 1:
Collect customer data from the existing market and identify the features that influence customer segmentation. Define the target variable as the customer segment (A, B, C, or D).
### STEP 2:
Remove irrelevant attributes, handle missing values, and encode categorical variables into numerical form. Split the dataset into training and testing sets.
### STEP 3:
Design a neural network classification model with suitable input, hidden, and output layers. Train the model using the training data to learn patterns for customer segmentation.
### STEP 4: Model Evaluation and Prediction
Evaluate the trained model using test data and use it to predict the customer segment for new customers in the target market.


## PROGRAM

### Name: SHRAVANI M
### Register Number: 212224230263

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,16)
        self.fc2=nn.Linear(16,8)
        self.fc3=nn.Linear(8,4)


    def forward(self, x):
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=self.fc3(x)
      return x
       

```
```python
# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.01)

```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
   for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      outputs = model(X_batch)
      loss = criterion(outputs, y_batch)
      loss.backward ()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```



## Dataset Information

<img width="1694" height="421" alt="image" src="https://github.com/user-attachments/assets/3f3f04e7-d87f-45c2-a5dd-3b96927f8815" />


## OUTPUT
### Confusion Matrix

<img width="880" height="581" alt="image" src="https://github.com/user-attachments/assets/5b5d7a6a-6b54-4a87-94da-e1c62f554efd" />




### Classification Report

<img width="1728" height="585" alt="image" src="https://github.com/user-attachments/assets/04904250-1bba-4166-a4ae-63b5adb6dfda" />



### New Sample Data Prediction

<img width="1090" height="386" alt="image" src="https://github.com/user-attachments/assets/b04373a3-c092-46b4-a785-1c2b2db73925" />




## RESULT

Thus neural network classification model is developded for the given dataset. 
