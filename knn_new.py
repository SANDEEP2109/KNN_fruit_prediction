import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

print("Fruit name prediction")
mass = float(input("Enter the mass of the fruit in grams: "))
width = float(input("Enter the width of the fruit in cm: "))
height = float(input("Enter the height of the fruit in cm: "))


fruits = pd.read_table("C:/Users/sande/Desktop/fruit_data_with_colors.txt")
# print(fruits.head())

lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

x = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

score = knn.score(x_test, y_test)
print(f"Accuracy: {score:.2f}")

new_data = [mass, width, height]
new_pred = knn.predict([new_data])
predicted_label = new_pred[0]
predicted_name = lookup_fruit_name[predicted_label]

print(f"Predicted Fruit: {predicted_name} (Label {predicted_label})")

new_row = pd.DataFrame([[new_data[0], new_data[1], new_data[2], predicted_label]],
                       columns=['mass', 'width', 'height', 'fruit_label'])

fruits_simple = fruits[['mass', 'width', 'height', 'fruit_label']]
fruits_updated = pd.concat([fruits_simple, new_row], ignore_index=True)

updated_path = "C:/Users/sande/Desktop/fruit_data_updated.txt"
fruits_updated.to_csv(updated_path, sep='\t', index=False)
print(f"New data added and saved to: {updated_path}")
