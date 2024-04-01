#PLOTS SOBRE LOS RESULTADOS. 
import pandas as pd
import matplotlib.pyplot as plt

predictions = pd.read_csv("hibrido//predictions.csv")


#plot 1: scatter 
plt.scatter(predictions['real'], predictions['final_ensemble'])
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title('Real vs Predicted Values')

#save the plot as hibridscatter.png
plt.savefig('hibrido//hibridscatter.png')

#-----

# plot 2: line plot
plt.plot(predictions['real'], label='Real Values')
plt.plot(predictions['final_ensemble'], label='Predicted Values')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Real vs Predicted Values')
plt.legend()

#save the plot as hibridline.png
plt.savefig('hibrido//hibridline.png')


#-----

#Plot 3: pintar los residuos en lineas y marcar los valores que estan fuera de los limites, dar los indices de estos valores.
# Calculate the residuals
residuals = predictions['real'] - predictions['final_ensemble']

# Calculate the mean and standard deviation of the residuals
mean_residual = residuals.mean()
std_residual = residuals.std()

# Calculate the upper and lower limits
upper_limit = mean_residual + 3 * std_residual
lower_limit = mean_residual - 3 * std_residual

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the residuals
ax.plot(residuals, label='Residuals')

# Plot the upper and lower limits
ax.axhline(upper_limit, color='r', linestyle='--', label='Upper Limit')
ax.axhline(lower_limit, color='r', linestyle='--', label='Lower Limit')

# Mark the values that are outside the limits
outliers = residuals[(residuals > upper_limit) | (residuals < lower_limit)]
outliers_index = outliers.index
ax.scatter(outliers_index, outliers, color='r', label='Outliers')

# Set the labels and title
ax.set_xlabel('Index')
ax.set_ylabel('Residuals')
ax.set_title('Residuals Plot')
ax.legend()



#save the plot as hibridresiduals.png

plt.savefig('hibrido//hibridresiduals.png')


print("outliers index:", outliers_index)

