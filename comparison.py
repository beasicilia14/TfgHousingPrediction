
#PRINT CONTENT OF FILE METRICS.CSV FROM ALL THE DIFFERENT MODELS 
import pandas as pd
import os

# Get the current working directory
cwd = os.getcwd()
print(cwd)

# Get the list of directories in the current working directory
directories = os.listdir(cwd)
print(directories)

# Create an empty dataframe to store the metrics
metrics_df = pd.DataFrame()

# Loop through the directories
for directory in directories:
    # Check if the directory is a directory and not a file
    if os.path.isdir(directory):
        # Get the path to the metrics.csv file
        metrics_file = os.path.join(directory, 'metrics.csv')
        # Check if the metrics.csv file exists in the directory
        if os.path.exists(metrics_file):
            # Read the metrics.csv file into a dataframe
            metrics = pd.read_csv(metrics_file)
            # Add a column to the dataframe to store the model name
            metrics['Model'] = directory
            # Append the metrics to the metrics_df
            metrics_df = metrics_df.append(metrics)

# Reset the index of the metrics_df
metrics_df = metrics_df.reset_index(drop=True)

# Print the metrics_df
print(metrics_df)

#Write into csv file called final_metrics.csv
metrics_df.to_csv("final_metrics.csv")

#plot the comparison of the models in a barchart, r2 , mape and mae 
import matplotlib.pyplot as plt

# Get the unique model names
models = metrics_df['Model'].unique()

# Create a figure and axis
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Loop through the metrics and plot the comparison
for i, metric in enumerate(['r2', 'mape', 'mae']):
    # Create a bar chart for the metric
    ax[i].bar(models, metrics_df.groupby('Model')[metric].mean())
    ax[i].set_title(metric.upper())
    ax[i].set_ylabel(metric.upper())
    ax[i].set_xlabel('Model')
    ax[i].set_xticklabels(models, rotation=45)

# Adjust the layout

plt.tight_layout()

plt.show()
