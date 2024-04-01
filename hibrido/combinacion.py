import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV


from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error


def get_metrics (y_real, y_pred): 
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    import numpy as np
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred)
    #put it in a df
    metrics = pd.DataFrame({'mse': [mse], 'rmse': [rmse], 'mae': [mae], 'r2': [r2], 'mape': [mape]})
    return metrics

cred = credentials.Certificate('claves.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

houses_ref = db.collection('cleaned_data')
houses = houses_ref.get()

# Create an empty dataframe
data = pd.DataFrame()

# Iterate over the houses and add each house's data to the dataframe
for house in houses:
    house_data = house.to_dict()
    data = data.append(house_data, ignore_index=True)


#### PROCESSING: 
data_numeric = data.select_dtypes(include=['float64', 'int64'])

data = data.drop(columns=["description", "propertyCode", "distance"], axis=1)
data_categorical = data.select_dtypes(include=['object'])

data_boolean = data.select_dtypes(include=['bool'])

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder()

# Fit and transform the categorical variables
encoded_data = encoder.fit_transform(data_categorical)

# Convert the encoded data to a dataframe
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(data_categorical.columns))

# Concatenate the encoded dataframe with the numeric dataframe
encoded_data = pd.concat([data_numeric, encoded_df, data_boolean], axis=1)


#CREACIÓN CONJUNTO TRAIN, TEST, VALIDACIÓN
X_train, X_test, y_train, y_test = train_test_split(encoded_data.drop("price", axis=1), encoded_data["price"], test_size=0.33, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)



#### FIRST MODEL: REGRESSION 
from sklearn.linear_model import LinearRegression

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(data_numeric.drop("price", axis=1), data_numeric["price"], test_size=0.33, random_state=42)
X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_train_reg, y_train_reg, test_size=0.3, random_state=42)

#X_train_num, X_test_num, y_train_num, y_test_num = train_test_split(data_numeric, data_numeric["price"], test_size=0.33, random_state=42)

#Creo modelo 
reg = LinearRegression().fit(X_train_reg, y_train_reg)

### VALIDATION. 
#Predigo para validation 
y_pred_val_reg = reg.predict(X_val_reg)
#Calculo error para luego ponderar.
mae_val_reg = mean_absolute_error(y_val_reg, y_pred_val_reg)

### CREO MODELO CON TRAIN Y VALIDATION COMO TRAIN
# Combinar los conjuntos de entrenamiento y validación
X_train_full_reg = np.concatenate((X_train_reg, X_val_reg))
y_train_full_reg = np.concatenate((y_train_reg, y_val_reg))
# Reentrenar el modelo con el conjunto de entrenamiento y validación combinado
reg.fit(X_train_full_reg, y_train_full_reg)

#Predigo para test
y_pred_reg = reg.predict(X_test_reg)


#### SECOND MODEL: RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
# best parameters: {'max_depth': 50, 'n_estimators': 200}
best_params_rf = {'max_depth': 50, 'n_estimators': 200}
rf = RandomForestRegressor(n_estimators=best_params_rf['n_estimators'], max_depth=best_params_rf['max_depth'])
rf.fit(X_train, y_train)

# Predigo para validation
y_pred_val_rf = rf.predict(X_val)
# Calculo error para luego ponderar.
mae_val_rf = mean_absolute_error(y_val, y_pred_val_rf)

# Combinar los conjuntos de entrenamiento y validación
X_train_full_rf = np.concatenate((X_train, X_val))
y_train_full_rf = np.concatenate((y_train, y_val))

# Reentrenar el modelo con el conjunto de entrenamiento y validación combinado
rf.fit(X_train_full_rf, y_train_full_rf)

# Predigo
y_pred_rf = rf.predict(X_test)



#### THIRD MODEL: GRADIENT BOOST
# Best Parameters: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 600}
best_params_gb = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 600}
xgb = XGBRegressor(n_estimators=best_params_gb['n_estimators'], max_depth=best_params_gb['max_depth'], learning_rate=best_params_gb['learning_rate'])
xgb.fit(X_train, y_train)

# Predigo para validation
y_pred_val_gb = xgb.predict(X_val)
# Calculo error para luego ponderar.
mae_val_gb = mean_absolute_error(y_val, y_pred_val_gb)

# Combinar los conjuntos de entrenamiento y validación
X_train_full_gb = np.concatenate((X_train, X_val))
y_train_full_gb = np.concatenate((y_train, y_val))

# Reentrenar el modelo con el conjunto de entrenamiento y validación combinado
xgb.fit(X_train_full_gb, y_train_full_gb)

# Predict the target variable
y_pred_gb = xgb.predict(X_test)


#wrtie in txt mae_val_gb, mae_val_rf, mae_val_reg   
with open('hibrido//info.txt', 'w') as f:
    print("mae_val_gb", mae_val_gb, file=f)
    print("mae_val_rf", mae_val_rf, file=f)
    print("mae_val_reg", mae_val_reg, file=f)

#Ponderación de los modelos
# Calcula las ponderaciones basadas en el inverso del MAE
weight_reg = 1 / mae_val_reg
weight_rf = 1 / mae_val_rf
weight_gb = 1 / mae_val_gb

# Normaliza las ponderaciones
total_weight = weight_reg + weight_rf + weight_gb
weight_reg /= total_weight
weight_rf /= total_weight
weight_gb /= total_weight

# Combina las predicciones ponderadas
y_pred = (y_pred_reg * weight_reg + y_pred_rf * weight_rf + y_pred_gb * weight_gb)

#Crear archivo csv con las predicciones, cada columna es un modelo.
predictions = pd.DataFrame({'regression': y_pred_reg, 'random_forest': y_pred_rf, 'gradient_boost': y_pred_gb, 'final_ensemble': y_pred, 'real': y_test})
predictions.to_csv('hibrido//predictions.csv', index=False)

#Calcular métricas
metrics = get_metrics(y_test, y_pred)
metrics.to_csv('hibrido//metrics.csv', index=False)
