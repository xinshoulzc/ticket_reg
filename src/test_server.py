import requests

headers={
"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36",
}

url = 'http://127.0.0.1:8081'
params = {'inputdir': 'datasets/debug', 'outputdir' :'datasets/dev', 'mode': 'price'}
response = requests.get(url=url, params=params, headers=headers).text
print(response)
