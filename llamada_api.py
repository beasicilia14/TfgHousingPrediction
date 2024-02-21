import requests
import json

# URL de la API
url = "https://api.idealista.com/3.5/es/search"

# Parámetros de la llamada a la API, variaremos únicamente el número de página
params = {
    "operation": "sale",
    "propertyType": "homes",
    "center": "40.416729,-3.703339",
    "distance": "3000",
    "locationId": "0-EU-ES-28-07-001-079",
    "maxItems": "50",
    "sinceDate": "Y",
    "order": "distance",
    "numPage": "11",
    "sort": "desc"

}

# Credenciales de acceso a la API en los headers
headers = {
    "Authorization": "Bearer escribiraqui",
}

# Realiza la llamada a la API
response = requests.request("POST", url, params=params, headers=headers)

# Verifica si la llamada fue exitosa (código de respuesta 200)
if response.status_code == 200:
    # La respuesta de la API está en formato JSON
    data = response.json()

    #guarda la respuesta en un archivo .json 
    with open(f'responses/response_{params["numPage"]}.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Response saved to response_{params['numPage']}.json")

else:
    # Si la llamada no fue exitosa, imprime el código de estado de la respuesta
    print("Error:", response.status_code)
