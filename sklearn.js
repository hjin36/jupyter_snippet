require(["nbextensions/snippets_menu/main"], function(snippets_menu) {
  console.log('Loading `snippets_menu` customizations from `custom.js`');
  var sklearn = {
    "name": "Sklearn",
    "sub-menu": [
      {
        "name": "Compatibility",
        "sub-menu": [
          {
            "name": "Python 2/3",
            "snippet": ["from __future__ import division, print_function, unicode_literals"
                     ]
        },
          {
            "name": "Ignore Warning",
            "snippet": ["import warnings",
                      "warnings.filterwarnings('ignore')",
                     ]
        }
      ]
    },
      {
        "name": "EDA",
        "sub-menu": [
          {
            "name": "Heat Map",
            "snippet": ["import seaborn as sns",
                        "from matplotlib import pyplot as plt",
                        "cor = df.corr()",
                        "top = cor.index",
                        "plt.figure(figsize=(20,20))",
                        "sns.heatmap(df[top].corr(),annot=True)"
                     ]
        }
      ]
    },
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
            "snippet": ["from sklearn.linear_model import LogisticRegression",
                        "logr = LogisticRegression(penalty='l1', C=100,random_state=42)"
                       ]
        },
          {
            "name": "K-Nearest Neighbors",
            "snippet": ["kn = KNeighborsClassifier(n_neighbors=9,p=2,",
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
        "name": "Unsupervised Learning",
        "sub-menu": [
          {
            "name": "K-means Clustering",
            "snippet": ["from sklearn.cluster import KMeans",
                      "from scipy.spatial.distance import cdist",
                      "kmeans = KMeans(n_clusters=4)\n#elbow method",
                      "distortions = []",
                      "K = range(1,10)",
                      "for k in K:",
                      "kmeanModel = KMeans(n_clusters=k).fit(X)",
                      "kmeanModel.fit(X)",
                      "distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])"
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
        },
          {
            "name": "F1 / Recall / Precision / Cohen Kappa",
            "snippet": ["from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, recall_score",
                      "# Replace the name for the corresponding metrics",
                      "print('Accuracy (out-of-sample): %.2f' % accuracy_score(y_test, y_pred))",
                      "print('Accuracy (in-sample): %.2f' % accuracy_score(y_train, y_pred_insample))"
                     ]
        },
          {
            "name": "Cross-Validation",
            "snippet": ["from sklearn.model_selection import cross_val_score",
                      "scores=cross_val_score(clf_lr, X, y, cv=5,scoring = 'f1')"
                     ]
        },
          {
            "name": "Grid Search",
            "snippet": ["from sklearn.model_selection import GridSearchCV, KFold, cross_val_score",
                        "inner_cv = KFold(n_splits=5, shuffle=True)",
			"outer_cv = KFold(n_splits=5, shuffle=True)",
                        "gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),",
                        "param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],",
                        "scoring='accuracy',",
                        "cv=inner_cv)",
			"gs = gs.fit(X,y)",
                        "print('Non-nested CV Accuracy: ', gs.best_score_)",
                        "print('Optimal Parameter: ', gs.best_params_)",
			"print('Optimal Estimator: ', gs.best_estimator_)",
                        "nested_score_gs = cross_val_score(gs, X=X, y=y, cv=outer_cv)",
                        "print('Nested CV Accuracy: ',nested_score_gs.mean(), ' +/- ', nested_score_gs.std())"
                     ]
        }
      ]
    }
    ]
  };
  snippets_menu.options['menus'].push(snippets_menu.default_menus[0]); // Start with the remaining "Snippets" menu
  snippets_menu.options['menus'].push(sklearn); 
  console.log('Loaded `snippets_menu` customizations from `custom.js`');
});
