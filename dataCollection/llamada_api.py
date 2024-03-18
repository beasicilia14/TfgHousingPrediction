import requests
import json

# URL de la API
url = "https://api.idealista.com/3.5/es/search"

# Parámetros de la llamada a la API, variaremos únicamente el número de página
params = {
    "operation": "sale",
    "propertyType": "homes",
    "center": "40.416729,-3.703339",
    "distance": "8000",
    "locationId": "0-EU-ES-28-07-001-079-06",
    "maxItems": "50",
    "sinceDate": "Y",
    "order": "distance",
    "numPage": "2",
    "sort": "desc"

}

# Credenciales de acceso a la API en los headers
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzY29wZSI6WyJyZWFkIl0sImV4cCI6MTcwOTMyNTYwOSwiYXV0aG9yaXRpZXMiOlsiUk9MRV9QVUJMSUMiXSwianRpIjoiZjM4Y2Q2NDUtYjkxMi00NmVkLTgwNWItODQ2NDE5YTU0ZWM0IiwiY2xpZW50X2lkIjoidmprY211eWY4NDNleHJvZ240MXA4cDlndGhjb2wwcjIifQ.4qwHz7mfgOfliKX5_Py1FpHLMROBuO6WA32_udT87Rc",
}

# Realiza la llamada a la API
response = requests.request("POST", url, params=params, headers=headers)

# Verifica si la llamada fue exitosa (código de respuesta 200)
if response.status_code == 200:
    # La respuesta de la API está en formato JSON
    data = response.json()

    #guarda la respuesta en un archivo .json 
    with open(f'totalResponses/responsesnew9/response_{params["numPage"]}.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Response saved to response_{params['numPage']}.json")

else:
    # Si la llamada no fue exitosa, imprime el código de estado de la respuesta
    print("Error:", response.status_code)
