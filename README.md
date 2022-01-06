# DMexam
## DATASET

KAGGLE.COM "customers_behavior.csv"

## Bibliothèques utilisés

pandas, numpy,
matplotlib.pyplot,
seaborn,
sklearn.preprocessing,
sklearn.preprocessing.StandardScaler,
statsmodels.stats.outliers_influence.variance_inflation_factor,
sklearn.model_selection.train_test_split,
sklearn.linear_model,
sklearn.metrics.r2_score,
sklearn.linear_model.LinearRegression

## Dataset cleaning:

1. Elimination des lignes ayant des valeurs null.
   df.dropna(inplace=True)

2. Conversion des types des variables en entiers.
   df = df.astype({
   "Administrative":"int", "Administrative_Duration":"int","Informational":"int","Informational_Duration":"int",
   "ProductRelated":"int","ProductRelated_Duration":"int", "BounceRates":"int", "ExitRates":"int", "PageValues":"int",
   "SpecialDay":"int"
   })

3. Convertir les variables catégorielles en des valeurs numeriques.
   df["Month"] = label_encoder.fit_transform(df["Month"])
   df["VisitorType"] = label_encoder.fit_transform(df["VisitorType"])
   df["Revenue"] = label_encoder.fit_transform(df["Revenue"])
   df["Weekend"] = label_encoder.fit_transform(df["Weekend"])

## Selection of Features and Labels

1. Séparation de la colonne a predire et le reste des colonnes du dataset.
   y = df[['Revenue']]
   X = df.drop(columns=['Revenue'])

2. Division de la colonne a predire et le rest du "dataset" en train et test
   X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)

## Regression

## Predection

    y_pred = model.predict(X_test)

## Evaluation

1. accuracy_score(y_test, y_pred)
2. roc_auc_score(y_test, y_pred)

## Optimization
