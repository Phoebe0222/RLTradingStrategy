######################################################################
# Use Yahoo Finance API to stream data, including stocks, options, and FX #
# Tutorial: https://docs.intrinio.com/tutorial/web_api               #
######################################################################



from __future__ import print_function
import requests
import time
import datetime as dt
import json 

today = dt.date.today()
today_timestamp = time.mktime(dt.datetime.strptime(str(today), '%Y-%m-%d').timetuple())

start_date = '2018-01-01'
start_timestamp = time.mktime(dt.datetime.strptime(str(start_date), '%Y-%m-%d').timetuple())


url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-historical-data"

querystring = {"frequency":"1d",
                "filter":"history",
                "period1":int(start_timestamp),
                "period2":int(today_timestamp),
                "symbol":"AMRN"}

headers = {
    'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com",
    'x-rapidapi-key': "e610553b4cmshb22dc6c3e9494f8p1467bejsn542357ece6c1"
    }

response = requests.request(
                            "GET", 
                            url, 
                            headers=headers, 
                            params=querystring)


data = response.json()