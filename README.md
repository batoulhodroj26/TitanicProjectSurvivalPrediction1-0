# TitanicProjectSurvivalPrediction1-0
This project builds a machine learning model to predict whether a passenger survived the Titanic disaster. It uses the famous Kaggle Titanic dataset and outputs a submission.csv file that can be uploaded to Kaggle for evaluation.
## 🔹 Project Overview

**Goal:** Build a machine learning model that predicts survival (`0` = No, `1` = Yes) of Titanic passengers based on features such as age, sex, ticket class, and more.  

**Dataset:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)  
Contains:
- `train.csv` → with labels (`Survived`)  
- `test.csv` → without labels (used for final predictions)

**Input Features Include:**

| Feature | Description |
|---------|-------------|
| `Pclass` | Ticket class (1st, 2nd, 3rd) |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | # of siblings/spouses aboard |
| `Parch` | # of parents/children aboard |
| `Fare` | Passenger fare |
| `Embarked` | Port of Embarkation (C, Q, S) |
| `Cabin`, `Ticket`, `Name` | Optional features for engineering |

**Target Variable:** `Survived` (0 or 1)

TitanicProject/
│
├── data/
│   ├── train.csv         # Input training dataset
│   ├── test.csv          # Input test dataset
│   └── .gitkeep          # Empty file to track empty folder
│
├── src/
│   ├── __init__.py       # empty, allows Python to recognize folder as a package
│   ├── load_data.py      # Loads train/test CSV files
│   ├── train_model.py    # Trains RandomForest model
│   └── predict.py        # Makes predictions & saves submission
│
├── main.py               # Main script to run the project
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
src/load_data.py
import pandas as pd
import os

def load_titanic_data():
    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


src/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from load_data import load_titanic_data

def train_model():
    train_df, _ = load_titanic_data()

    # Preprocessing
    df = train_df.copy()
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                'Embarked_Q', 'Embarked_S']
    X = df[features]
    y = df['Survived']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    print(f"Model Accuracy: {accuracy_score(y_val, y_pred):.4f}")

    return model


src/predict.py
import pandas as pd
from load_data import load_titanic_data

def make_predictions(model, test_df):
    df = test_df.copy()
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # Ensure columns match training
    for col in ['Embarked_Q', 'Embarked_S']:
        if col not in df.columns:
            df[col] = 0

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                'Embarked_Q', 'Embarked_S']

    X_test = df[features]
    predictions = model.predict(X_test)
    return predictions

def save_submission(test_df, predictions):
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv('data/submission.csv', index=False)
    print("✅ Submission saved to data/submission.csv")


main.py
import sys
import os

# Add src folder to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import modules from src
from load_data import load_titanic_data
from train_model import train_model
from predict import make_predictions, save_submission

def main():
    print("🔍 Loading data...")
    train_df, test_df = load_titanic_data()
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}\n")

    print("📚 Training model...")
    model = train_model()
    print("✅ Model trained!\n")

    print("📝 Making predictions...")
    predictions = make_predictions(model, test_df)
    save_submission(test_df, predictions)

    print("\n✅ All done! Check data/submission.csv for results.")

if __name__ == "__main__":
    main()


requirements.txt
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost


3️⃣ Data


Place train.csv and test.csv inside data/ folder.



4️⃣ How It Works


main.py runs the workflow:


Loads data (load_data.py)


Preprocesses & trains model (train_model.py)


Makes predictions & saves CSV (predict.py)




Random Forest classifier is used.


Output saved as:


data/submission.csv

Format:
PassengerId,Survived
892,0
893,1
894,0
...


5️⃣ Running the Project
python main.py

Expected output in terminal:
🔍 Loading data...
Train shape: (891, 12), Test shape: (418, 11)

📚 Training model...
Model Accuracy: 0.78xx
✅ Model trained!

📝 Making predictions...
✅ Submission saved to data/submission.csv

✅ All done! Check data/submission.csv for results.


6️⃣ GitHub Setup Tips


Add a .gitignore for Python:


__pycache__/
*.pyc
*.pyo
*.pyd
.env
venv/
data/submission.csv
Output
🔍 Loading data...
Train shape: (891, 12), Test shape: (418, 11)

📚 Training model...
E:\TitanicProject\src\train_model.py:12: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
  df['Age'].fillna(df['Age'].median(), inplace=True)
E:\TitanicProject\src\train_model.py:13: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
  df['Fare'].fillna(df['Fare'].median(), inplace=True)
E:\TitanicProject\src\train_model.py:14: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
  df['Embarked'].fillna('S', inplace=True)
Model Accuracy: 0.7989
✅ Model trained!

📝 Making predictions...
E:\TitanicProject\src\predict.py:6: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
  df['Age'].fillna(df['Age'].median(), inplace=True)
E:\TitanicProject\src\predict.py:7: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
  df['Fare'].fillna(df['Fare'].median(), inplace=True)
✅ Submission saved to data/submission.csv

✅ All done! Check data/submission.csv for results.

