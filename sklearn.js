{
  "name": "scikit-learn",
  "sub-menu": [
    {
      "name": "Processing",
      "sub-menu": [
        {
          "name": "Train Test Split",
          "snippet": ["from sklearn.model_selection import train_test_split",
                      "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)"
                     ]
        },
        {
          "name": "Standard Scale",
          "snippet": ["from sklearn.preprocessing import StandardScaler",
                      "sc = StandardScaler()",
                      "sc.fit(X)",
                      "X = sc.transform(X)"
                     ]
        }
      ]
    },
    {
      "name": "Feature Engineering",
      "sub-menu": [
        {
          "name": "L1 Feature Selection",
          "snippet": ["from sklearn.linear_model import LogisticRegression",
                      "lr = LogisticRegression(penalty='l1', C=1.0)",
                      "lr.fit(X_train, y_train)",
                      "print('Training accuracy:', lr.score(X_train, y_train))",
                      "print('Test accuracy:', lr.score(X_test, y_test))"
                     ]
        }
      ]
    },
    {
      "name": "Classification",
      "sub-menu": [
        {
          "name": "Decision Tree",
          "snippet": ["dt = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5",
                      "min_impurity_decrease=0.0, min_samples_leaf=6, min_samples_split=2,random_state=42)"
                     ]
        },
        {
          "name": "Logistic Regression",
          "snippet": ["logr = LogisticRegression(penalty='l1', C=100,random_state=42)"]
        },
        {
          "name": "K-Nearest Neighbors",
          "snippet": ["kn = KNeighborsClassifier(n_neighbors=9,p=2",
                      "metric='minkowski',weights='uniform')"
                     ]
        }
      ]
    },
    {
      "name": "Regression",
      "sub-menu": [
        {
          "name": "Linear Regression",
          "snippet": ["from sklearn.linear_model import LinearRegression",
                      "lr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)"
                     ]
        }
      ]
    },
    {
      "name": "Evaluation",
      "sub-menu": [
        {
          "name": "Confusion Matrix",
          "snippet": ["from sklearn.metrics import confusion_matrix",
                      "cnf_matrix = confusion_matrix(y_test, y_pred)"
                     ]
        },
        {
          "name": "Classification Report",
          "snippet": ["from sklearn.metrics import classification_report",
                      "print(classification_report(y_test, y_pred))"
                     ]
        }
      ]
    }
    ]
}
