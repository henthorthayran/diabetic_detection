import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

!wget https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv


df = pd.read_csv('diabetes.csv')
df.head()



x=df.drop('Outcome',axis=1).values
y=df['Outcome'].values

# import train test split and all pytorch library and nn

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0 )

#  change data  in to tensor

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

df.shape

class ANN_Model(nn.Module):
  def __init__(self,input_features=8,hidden1=50,hidden250,out_features=2):
    super().__init__()
    self.fc1 = nn.Linear(input_features,hidden1)
    self.fc2 = nn.Linear(hidden1,hidden2)
    self.out = nn.Linear(hidden2,out_features)
  def forward(self,x):
    out = F.relu(self.fc1(x))
    out = F.relu(self.fc2(out))
    out = self.out(out)
    return out

torch.manual_seed(20)
model = ANN_Model()

# backpropagation using optamiser adam

#Backward Propogation -- adam optimizer
loss_function= nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)## lr =learningrate

#train the model epochs=500

epochs = 500
final_losses = []
for i in range(epochs):
  i=i+1
  y_pred = model.forward(x_train)
  loss = loss_function(y_pred,y_train)
  final_losses.append(loss)
  if i%1 == 1:
    print("Epoch number: {} and the loss : {}".format(i,loss.item()))
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# plot  the graph epoch Vs loss

import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(range(epochs), [loss.detach().numpy() for loss in final_losses])

plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Epoch vs Loss')
plt.show()

prediction=[]
with torch.no_grad():
  for i  ,data in enumerate(x_test):
   y_pred = model(data)
   prediction.append(y_pred.argmax().item())
   print(y_pred.argmax().item())


# compare prediction  with real data

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,prediction)
print(cm)
print(accuracy_score(y_test,prediction))


# plote the cofusion matrix as form of true posetive , fulse posetive

import matplotlib.pyplot as plt
# Assuming 'cm' is your confusion matrix calculated previously
# cm = confusion_matrix(y_test,prediction)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')

# Extracting True Positives, True Negatives, False Positives, False Negatives
tn, fp, fn, tp = cm.ravel()

print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")

plt.show()



# save the model

torch.save(model, 'diabetes.pt')

from torch.serialization import add_safe_globals

# Register the custom model class for unpickling
add_safe_globals([ANN_Model])

loaded_model = torch.load('diabetes.pt', weights_only=False)
loaded_model.eval()


list(df.iloc[0,:-1])

list1=[5.0, 140.0, 60.0, 35.0, 1.0, 34.6, 0.627, 40.0]

new_data=torch.tensor(list1)

#predict new_data

with torch.no_grad():
  predicted_output = loaded_model(new_data)
  predicted_class = torch.argmax(predicted_output).item()

print(f"The predicted class for the new data is: {predicted_class}")
