from requests import Request, Session
import pandas as pd
import json
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
from datetime import datetime, timedelta, timezone
import numpy as np
import tensorflow as tf
from keras import models, Sequential, layers, optimizers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

class CryptoData:
   # intialize api key as instance variable for API calls
   def __init__(self, api_key):
      self.api_key = api_key

   # fetches the exchange id of the exchange that the user provides. 
   def fetch_exchange_id(self):
      # prompts user for exchange
      user_exchange = input('Enter the crypto exchange you would like to look at: ')
      # api url for exchange ids
      url = "https://rest.coinapi.io/v1/exchanges"
      # paramteres and headers to be passed to api call
      payload={}
      headers = {
      'Accept': 'text/plain',
      'X-CoinAPI-Key': self.api_key
      }

      # initialize new session class for api call
      session = Session()
      session.headers.update(headers)

      # retreive data from api call
      try:
         response = session.get(url, params=payload)
         data = json.loads(response.text)
      except(ConnectionError, Timeout, TooManyRedirects) as e:
         print(e)

      # stores data into variable after data is normalized to dataframe
      data = pd.json_normalize(data)

      # fetches the exact exchange id that the user provided from the dataframe of all exchange ids 
      exchange_id = data['exchange_id'].loc[data['name'].str.lower() == user_exchange]

      # returns the exchange id
      return exchange_id

   # fetches the asset id of the crypto currency the user wants to observe
   def fetch_asset_id(self):
      # Prompts the user for a crypto currency
      crypto = input("Enter Cryptocurrency one by one to be analyzed: ")
      # api url for asset ids
      url = "https://rest.coinapi.io/v1/assets"
      # paramteres and headers to be passed to api call
      payload={}
      headers = {
      'Accept': 'text/plain',
      'X-CoinAPI-Key': self.api_key
      }
      # initialize new session class for api call
      session = Session()
      session.headers.update(headers)
      # retreive data from api call
      try:
         response = session.get(url, params=payload)
         data = json.loads(response.text)
      except(ConnectionError, Timeout, TooManyRedirects) as e:
         print(e)
      # stores data into variable after data is normalized to dataframe
      data = pd.json_normalize(data)

      # fetches the exact asset id that the user provided from the dataframe of all asset ids as a Series type
      asset_id = data['asset_id'].loc[data['name'].str.lower() == crypto]

      # fetches the Series of string types of the initial series
      asset_id = asset_id.astype(str).values

      # returns the first index of the series, which is the appropriate asset id
      return asset_id[0]
   

# Fetches the symbol id correlated to the cryptocurrency provided by the user
   def fetch_symbol_id(self):
      # retrieves the exchange id
      exchange_id = self.fetch_exchange_id()
      # retrieves the asset id of the currency
      asset_id = self.fetch_asset_id()
      # api url for symbol ids
      url = "https://rest.coinapi.io/v1/symbols"

      # parameters and headers to be passed to api call (we will filter the data by the asset and exchange id provided)
      payload={
         'filter_exchange_id': exchange_id,
         'filter_symbol_type': 'SPOT',
      }
      headers = {
      'Accept': 'text/plain',
      'X-CoinAPI-Key': self.api_key
      }
      # initialize new session class for api call
      session = Session()
      session.headers.update(headers)
      # retreive data from api call
      try:
         response = session.get(url, params=payload)
         data = json.loads(response.text)
      except(ConnectionError, Timeout, TooManyRedirects) as e:
         print(e)
      
      # stores data into variable after data is normalized to dataframe
      data = pd.json_normalize(data)

      # filters the dataframe by the type of currency being reffered to (in this case the U.S. dollar)
      data = data.loc[data['asset_id_quote'] == 'USD']
      
      # fetches the symbol id of the cryptocurrency based off of the asset id as a Series object
      raw_id = data['symbol_id'].loc[data['asset_id_base'] == asset_id]

      # returns the series where the values are strings
      raw_id = raw_id.astype(str).values

      # returns the first index of the series, which is the appropriate symbol id
      return raw_id[0]

   # Returns all historical data in the last year correlating to the cryptocurrency provided
   def fetch_crypto_data(self):
      # retrives the symbol_id of specified cryptocurrency
      symbol_id = self.fetch_symbol_id()
      # api url for historical data on crypto currency given the symbol id of the currency
      url = f"https://rest.coinapi.io/v1/ohlcv/{symbol_id}/history"

      # Gets the current date
      end_time_raw = datetime.now(timezone.utc)

      # Gets the date from a year ago
      start_time_raw = end_time_raw - timedelta(days=365)

      # formats the end time to iso 8601 for api call
      end_time = end_time_raw.strftime('%Y-%m-%dT%H:%M:%S')

      # formats the start time to iso 8601 for api call
      start_time = start_time_raw.strftime('%Y-%m-%dT%H:%M:%S')

      # Parameters and headers passed to api call, we will record data for each day
      payload={
         'period_id': '1DAY',
         'time_start': start_time,
         'time_end': end_time,
         'limit' : '10000'
      }
      headers = {
      'Accept': 'text/plain',
      'X-CoinAPI-Key': self.api_key,
      }
      
      # initialize new session class for api call
      session = Session()
      session.headers.update(headers)
      # retreive data from api call
      try:
         response = session.get(url, params=payload)
         data = json.loads(response.text)
      except(ConnectionError, Timeout, TooManyRedirects) as e:
         print(e)

      # stores data into variable after data is normalized to dataframe
      data = pd.json_normalize(data)

      # Adds a column to the dataframe for the symbol id of the data
      data['symbol_id'] = symbol_id

      # Returns all historical data about the cryptocurrency as a pandas dataframe
      return pd.DataFrame(data)
   


      


