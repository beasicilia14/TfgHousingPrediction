import pandas as pd 

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


#apply function get_metrics to the columns of predictions_wo_outs.csv final_ensemble and real
predictions = pd.read_csv("hibrido//predictions_wo_outs.csv")

metrics = get_metrics(predictions['real'], predictions['final_ensemble'])

#save the metrics into a csv file called metrics_wo_outs.csv
metrics.to_csv("hibrido//metrics_wo_outs.csv")

