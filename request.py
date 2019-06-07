import requests

# URL
url = 'http://localhost:5000/api'

# Change the value of experience that you want to test
stri=input("Enter news: ")
#stri=[my]
print(type(stri))
#exp=float(exp1)
r = requests.post(url,json={'stri':stri})
print(r.json())