#SCRIPT TO GET FINAL VERSION OF DATASET. 
#This script is used to clean the data collected from the API and store it in a new collection in Firebase.

import ast
import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


#Load all the collected data 
cred = credentials.Certificate('claves.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
houses_ref = db.collection('houses')
houses = houses_ref.get()

# Create an empty dataframe
data = pd.DataFrame()

# Iterate over the houses and add each house's data to the dataframe
for house in houses:
    house_data = house.to_dict()
    data = data.append(house_data, ignore_index=True)

#Drop columns that are not useful for the model
data.drop(columns=["description", "propertyCode", "distance"], inplace=True)

#### Creating the typology variable 
def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return s  # return the original string if it can't be parsed

data['detailedType'] = data['detailedType'].apply(safe_literal_eval)
# Extract the 'typology' and 'subTypology' values
data['typology'] = data['detailedType'].apply(lambda x: x.get('typology'))
data['subTypology'] = data['detailedType'].apply(lambda x: x.get('subTypology'))

# Modify 'typology' based on 'subTypology'
data['typology'] = data.apply(lambda row: row['typology'] if pd.isnull(row['subTypology']) else row['typology'] + ' ' + row['subTypology'], axis=1)
# Drop the 'subTypology' column
data = data.drop(columns=['subTypology'])
data = data.drop(columns=['detailedType'])

#### Parking
# Create columns 
data['hasParking'] = data['parkingSpace'].apply(lambda x: 'yes' if x != 'nan' else 'no')
#drop parkingSpace column
data = data.drop(columns=['parkingSpace'])

#### Creating final dataset
#Remove id variables: thumbnail, externalReference, url, hasVideo, hasStaging, has460, has3DTour, labels
data = data.drop(columns=['thumbnail', 'externalReference', 'url', 'hasVideo', 'hasStaging', 'has360', 'has3DTour', 'labels'])

#Remove specific variables: showAddress, suggestedTexts, numPhotos, description, operation, address
data = data.drop(columns=['showAddress', 'suggestedTexts', 'numPhotos', 'operation', 'address', 'highlight', 'newDevelopmentFinished', 'topNewDevelopment', 'topPlus','hasPlan', 'propertyType'])

data = data.dropna()

#Store the data in firebase as a new collection called cleaned_data
cleaned_data_ref = db.collection('cleaned_data')

# Convert the dataframe to a dictionary
data_dict = data.to_dict(orient='records')

# Upload the data

for row in data_dict:
    cleaned_data_ref.add(row)