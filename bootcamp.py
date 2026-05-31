import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load customer dataset using Pandas
print("1. Loading data from Excel...")
file_path = "customer_reatil.csv.xlsx"
df = pd.read_excel(file_path)

# Take a random sample to make the code run fast!
print("   -> Taking a random sample of 20,000 rows...")
df = df.sample(n=20000, random_state=42)

# Filter to use only the features mentioned in the document
df = df[['Quantity', 'UnitPrice', 'Country']]

# 2. Handle missing values
print("2. Cleaning missing values...")
df = df.dropna()

# 3. Encode categorical columns using LabelEncoder
print("3. Encoding 'Country' into numbers...")
le = LabelEncoder()
df['Country_Encoded'] = le.fit_transform(df['Country'])

# 4. Visualize customer data using Matplotlib 
print("4. Generating Customer Distribution Graph... (Close the graph to continue!)")
plt.figure(figsize=(10, 6))
df['Country'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Countries by Customer Distribution')
plt.xlabel('Country')
plt.ylabel('Number of Transactions')
plt.tight_layout()
plt.show() 

# 5. Split data into training and testing sets
print("5. Splitting data into Training and Testing sets...")
X = df[['Quantity', 'UnitPrice']]
y = df['Country_Encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- NEW: Scale the features so the math runs instantly! ---
print("   -> Scaling features for faster training...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Logistic Regression model (Using SCALED data)
print("6. Training Logistic Regression model...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# 7. Train Decision Tree model (Trees don't need scaling, but it doesn't hurt)
print("7. Training Decision Tree model...")
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train_scaled, y_train)

# 8. Train KNN model (KNN desperately needs scaled data to work well!)
print("8. Training K-Nearest Neighbors (KNN) model...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 9. Evaluate models using Accuracy and Confusion Matrix
print("\n--- Final Model Evaluation ---")
models = {'Logistic Regression': log_reg, 'Decision Tree': dtree, 'KNN': knn}
accuracies = {}

for name, model in models.items():
    # Make sure to test on the scaled data!
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} Confusion Matrix:\n{cm}\n")

# 10. Compare model performances using graphs
print("10. Generating Model Accuracy Comparison Graph...")
plt.figure(figsize=(8, 5))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'orange'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Machine Learning Models')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1)

for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.02, str(round(v, 4)), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()