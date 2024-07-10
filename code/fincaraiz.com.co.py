import csv
import requests


def api_call(filter,offset=0,limit=100):


    headers = {
        'Host': 'kong.fincaraiz.com.co',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'sec-ch-ua-platform': '"Windows"',
        'sec-ch-ua-mobile': '?0',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'content-type': 'application/json',
        'accept': '*/*',
        'origin': 'https://www.fincaraiz.com.co',
        'sec-fetch-site': 'same-site',
        'sec-fetch-mode': 'cors',
        'sec-fetch-dest': 'empty',
        'referer': 'https://www.fincaraiz.com.co/',
        'accept-language': 'en-IN,en;q=0.9,zh-TW;q=0.8,zh;q=0.7,ml;q=0.6,hi;q=0.5',
    }

    json_data = {
        'filter': filter,
        'fields': {
            'exclude': [],
            'facets': [],
            'include': [
            ],
            'limit': limit,
            'offset': offset,
            'ordering': [],
            'platform': 40,
            'with_algorithm': True,
        },
    }

    response = requests.post('https://kong.fincaraiz.com.co/api/v1/listings/search', headers=headers, json=json_data)
    jsn = response.json()



    try:
        for hit in jsn['hits']['hits']:
            source = hit['_source']['listing']

            dataset = {}

            for dataset_key in ['neighbourhoods', 'cities', 'states', 'countries']:
                try:
                    dataset[f"location_{dataset_key}"] = source['locations'][dataset_key][0]['name']
                except:
                    dataset[f"location_{dataset_key}"] = None

            dataset['property_types'] = ",".join([x['name'] for x in source['property_type']])
            dataset['Transaction type'] = ",".join([x['name'] for x in source['offer']])
            dataset['State'] = source['condition']['name']

            dataset['Price'] = str(source['price'])

            dataset['Built meters'] = str(source['area'])
            dataset['Square meters'] = str(source['price_m2'])
            dataset['Private square meters'] = str(source['living_area'])
            dataset['Estrato'] = str(source['stratum']['name'])
            dataset['Rooms'] = str(source['rooms']['name'])
            dataset['Bathrooms'] = str(source['baths']['name'])
            dataset['Parking'] = str(source['garages']['name'])
            dataset['Internal floors'] = str(source['interior_floors'])
            try:
                dataset['First picture'] = source['media']['photos'][0]['list'][0]['image']['full_size']
            except:
                dataset['First picture'] = None

            dataset['Characteristics list'] = " , ".join([x['name'] for x in source['categories']])
            dataset['Antiquity'] = source['age']['name']

            yield dataset
    except:
        pass



    if offset == 0:

        total = jsn['hits']['total']['value']
        if total > limit:
            for offset in range(limit,total,limit):
                yield from api_call(filter,offset,limit=limit)


if __name__ == '__main__':
    #https://www.fincaraiz.com.co/apartamentos/venta/antioquia?pagina=1
    apartamentos_venta_antioquia = {
        'offer': {
            'slug': [
                'sell',
            ],
        },
        'property_type': {
            'slug': [
                'apartment',
            ],
        },
        'locations': {
            'states': {
                'slug': [
                    'state-colombia-05-antioquia',
                    'colombia-antioquia',
                ],
            },
        },
    }

    #https://www.fincaraiz.com.co/casas/venta/santander?pagina=1&usado=true

    casas_venta_santander = {
        'offer': {
            'slug': [
                'sell',
            ],
        },
        'is_new': False,
        'property_type': {
            'slug': [
                'house',
            ],
        },
        'locations': {
            'states': {
                'slug': [
                    'state-colombia-68-santander',
                    'colombia-santander',
                ],
            },
        },
    }

    row_list = []
    row_list += list(api_call(casas_venta_santander))
    row_list+=list(api_call(apartamentos_venta_antioquia))

    with open('fincaraiz.csv','w',newline='',encoding='utf-8-sig') as f:
        csv_writer = csv.DictWriter(f,fieldnames=row_list[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(row_list)


