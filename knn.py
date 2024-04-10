import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
spam = pd.read_csv('spam.csv')
X = spam['v2']
y = spam['v1']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert email text into a matrix of token counts using CountVectorizer
cv = CountVectorizer()
X_train_features = cv.fit_transform(X_train)
X_test_features = cv.transform(X_test)

# Instantiate a kNN classifier with 5 neighbors
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# Fit the model to the training data
knn.fit(X_train_features, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test_features)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using seaborn's heatmap function
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
