import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Hyper-parameters
EPOCHS = 20
LOSS = 'binary_crossentropy'

data = pd.read_csv('../data/Customer_Churn_processed.csv')
X = data.drop(columns=['LEAVE'])
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = data['LEAVE']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=True)

model = Sequential()
model.add(Dense(64, input_dim=11, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss=LOSS, optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=EPOCHS)

_, final_accuracy = model.evaluate(X_test, y_test)
print(final_accuracy)

# plot
plt.plot(history.history['accuracy'])
plt.title('Accuracy over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()