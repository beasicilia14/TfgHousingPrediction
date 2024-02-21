import firebase_admin
from firebase_admin import credentials, firestore
import json

# Step 2: Initialize Firestore
# Make sure you have downloaded the service account key and replace 'path/to/serviceAccountKey.json' with its path
cred = credentials.Certificate('claves.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

# Step 3: Read JSON data from file
for i in range(1, 11):
    with open(f'responses/response_{i}.json', 'r', encoding='utf-8') as file:
        json_data = file.read()
        
    data = json.loads(json_data)

    # Step 4: Upload the parsed data to Firestore
    element_list = data['elementList']

    for doc_data in element_list:
        doc_ref = db.collection('houses').document()
        doc_ref.set(doc_data)
        print(f'Document {doc_ref.id} uploaded successfully.')

print('All documents uploaded successfully.')