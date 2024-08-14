import pandas as pd
import numpy as np
import tensorflow as tf
from keras import models, Sequential, layers, optimizers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from cryptodata import CryptoData
import datetime




class crypto_predictions:
      # Initialize the historical data of the cryptocurrency as an instance variable
      def __init__(self, data):
         self.data = data
      
      # Converts the parameter "s" (string type) to a datetime object
      def str_to_datetime(self, s):
         # Splits the date string to get rid of the "T00:0000"
         date = s.split("T")[0]
         # Splits the date into year, month, day
         split = date.split('-')
         # Assigns each part of "split" to the variables
         year, month, day = int(split[0]), int(split[1]), int(split[2])
         # returns these in datetime format
         return datetime.datetime(year=year, month=month, day=day)
      
      # Fetches the price data of the cryptocurrency
      def create_price_df(self):
         # Takes a copy of the dataframe
         temp_df = self.data.copy()
         # Initializes a new dataframe which contains the time period close and the price close of each day
         price_df = temp_df[['time_period_end', 'price_close']]
         # Iterates through the entire dataframe to turn each string type date to a datetime object
         for i in range(len(price_df)):
            price_df.at[i, 'time_period_end'] = self.str_to_datetime(price_df.at[i, 'time_period_end'])
         
         # Makes the index of the dataframe the time in which each day ended 
         price_df.index = price_df.pop('time_period_end')
         # returns the price dataframe
         return price_df
      
      # Returns a windowed dataframe of the price dataframe in which the last three prices prior to the actual price are inserted as targets in each row
      def df_to_windowed_df(self):
         # Creates the price dataframe
         dataframe = self.create_price_df()
         # n represents the number of time steps taken (in this case: 3 days)
         n = 3
         # This gets the first date from where to start(I started where on the 4th day so I could use the first three days in my first window)
         first_date = dataframe.index[3].strftime('%Y-%m-%d')
         # This gets the last date
         last_date = dataframe.index[-1].strftime('%Y-%m-%d')

         # Converts the first date to a datetime object
         first_date = self.str_to_datetime(first_date)
         # Converts the last date to a datetime object
         last_date = self.str_to_datetime(last_date)
         
         # We will start by initializing our target day to be our first date
         target_date = first_date
  
         # Creates two new lists to store the dates and X, Y sets
         dates = []
         X, Y = [], []

         last_time = False

         # While loop generates the windowed dataframe
         while True:
            # Grabs a subset of the current frame which contains the last n+1 rows up to the current target date
            df_subset = dataframe.loc[:target_date].tail(n+1)
    
            # This throws an error if the window size to too large from the given target date and breaks the loop
            if len(df_subset) != n+1:
               print(f'Error: Window of size {n} is too large for date {target_date}')
               return

            # Converts the price column to a numpy array
            values = df_subset['price_close'].to_numpy()
            # This seperates the values from our inputs and expected outputs
            x, y = values[:-1], values[-1]
            
            # Appends the target date to our date list
            dates.append(target_date)
            # Appends the input set to the X list
            X.append(x)
            # Appends the output set to the Y list
            Y.append(y)

            # The following is used to grab the next target date

            # First, we grab the next day a week from the target date
            next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
            # Next, we convert this to a string and grab the last value from the top of the values list
            next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
            # Then, we need to seperate split the string to get the correct format (similar to the str_to_datetime() function above)
            next_date_str = next_datetime_str.split('T')[0]
            year_month_day = next_date_str.split('-')
            year, month, day = year_month_day
            next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
    
            # If this is the last date, the loop will end 
            if last_time:
               break
    
            # Otherwise, the target date will be whatever the next date generated above is
            target_date = next_date

            # If the target date and last date are the same, then the loop will break
            if target_date == last_date:
               last_time = True
    
         # Initializes a empty data frame
         ret_df = pd.DataFrame({})
         # Creates a new column where the dates are in the Target date column
         ret_df['Target Date'] = dates
  
         # Puts the input set as an numpy array
         X = np.array(X)
         # Creates a new column for each target and adds each previous price to each target window column
         for i in range(0, n):
            X[:, i]
            ret_df[f'Target-{n-i}'] = X[:, i]
  
         # Creates target column with the expected outputs
         ret_df['Target'] = Y
         
         # Returns the finished windowed dataframe
         return ret_df
      
      # 
      def windowed_df_to_date_X_y(self):
         # Retreives the windowed dataframe
         windowed_dataframe = self.df_to_windowed_df()
         # Converts the windowed dataframe to a numpy array
         df_as_np = windowed_dataframe.to_numpy()
         
         # Extracts everything from up to the first column from the numpy array 
         dates = df_as_np[:, 0]

         # Extracts everything from the numpy array except the first and last columns
         middle_matrix = df_as_np[:, 1:-1]
         # Reshapes the matrix into a 3D array. the shape becomes (number of dates, number of features, 1)
         # This will make it compatible with our model which expects a 3D shape
         X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

         # Extracts the last column which is the target variable labels
         Y = df_as_np[:, -1]

         # Returns the dates and input/outputs sets in corrected format and data type
         return dates, X.astype(np.float32), Y.astype(np.float32)
      

      # Splits up the adjusted 3D matrix into training, validation, and testing sets
      def split_df_to_sets(self):
         # Gets the dates, X, Y from the previous function
         dates, X, Y = self.windowed_df_to_date_X_y()
         # Seperates the dates into percentiles to distribute to our model's sets
         q_80 = int(len(dates) * .8)
         q_90= int(len(dates) * 0.9)
         # I used the Min and Max Scaler to normalize the data for the model to learn at a better rate
         scaler = MinMaxScaler()
         # Uses the dimensions of the X input to get the samples, timesteps, and features
         samples, timesteps, features = X.shape

         # I had to reshape the x in order to scale the inputs then reshape it back to the corrected format for model learning
         x_reshaped = X.reshape(-1, features)

         x_scaled = scaler.fit_transform(x_reshaped)

         x_scaled = x_scaled.reshape(samples, timesteps, features)

         # We will split the first 80% of our data for our training sets, 10% to the validation sets, and the last 10% to the testing sets

         dates_train, X_train, y_train = dates[:q_80], x_scaled[:q_80], Y[:q_80]
         dates_val, x_val, y_val = dates[q_80:q_90], x_scaled[q_80:q_90], Y[q_80:q_90]
         dates_test, X_test, y_test = dates[q_90:], x_scaled[q_90:], Y[q_90:]

         # Create labels for the dictionary we will return which return all of the following

         labels = ['dates_train', 'X_train', 'y_train', 'dates_val', 'x_val', 'y_val', 'dates_test', 'X_test', 'y_test']

         # Initialize an empty dictionary

         set_dict = dict.fromkeys(labels, [])

         # Updates our dictionary using the values from above
         set_dict.update({
            'dates_train': dates_train,
            'X_train': X_train,
            'y_train': y_train,
            'dates_val': dates_val,
            'x_val': x_val,
            'y_val': y_val,
            'dates_test': dates_test,
            'X_test': X_test,
            'y_test': y_test
         })

         # Returns the dictionary

         return set_dict
      
      # This function is used to build and train our LSTM model
      def launch_LSTM_model(self, X_train, y_train, x_val, y_val, dates_train):
         # I used the basic sequential class and passed one LSTM layer for the time series analysis and two hidden density layers and dropout
         # of 0.2 between each hidden layer. The final layer is expected to only have one output. 
         model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(256),
                    layers.Dropout(0.2),
                    layers.Dense(128, activation='relu'),
                    layers.Dropout(0.2),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(1)])

         # This will compile our model using mean square error to calculate loss and and learning rate of 0.001 to not rush the learning rate
         # of the machine
         model.compile(loss='mse', 
              optimizer=optimizers.Adam(learning_rate=0.001),
              metrics=['mse'])

         # This trains our data using 250 cycles and using validation sets
         model.fit(X_train, y_train, epochs=250, validation_data=(x_val, y_val))
         # We will store our predicitons for the training set in this variable
         train_predictions = model.predict(X_train).flatten()

         # This will plot our training predictions and observations to visualize the results of training the model
         plt.plot(dates_train, train_predictions)
         plt.plot(dates_train, y_train)
         plt.legend(['Training Predictions', 'Training Observations'])

         # returns the trained model
         return model
      
      # We will use this function to make predictions on days in the future on the price of crypto currency
      def make_future_predictions(self, model, X_test, dates_test):
         # Initializes an empty list to store predicitions during the recursive process
         recursive_predictions = []
         # Extracts the last seven dates and stores them in this new varialbe
         recursive_dates = dates_test[-7:]
         # Retrives the corresponding data to the last seven days
         last_window  = X_test[-7]
         for target_date in recursive_dates:
            # Uses the model to predict the next value based on the current last_window
            next_prediction = model.predict(np.array([last_window])).flatten()
            # Appends this prediction to the prediction list
            recursive_predictions.append(next_prediction)
            # Creates a new list by taking all elements of "last_window" except for the first one to shift the window forward
            new_window = list(last_window[1:])
            # Appends the next prediction to the new_window list
            new_window.append(next_prediction)
            # Converts "new_window" back to a numpy array so it can be used as input for the next prediciton
            new_window = np.array(new_window)
            # Updates "last_window" to be "new_window" making it the current window for the next iteration in the loop
            last_window = new_window
         
         # Creates a plot visulization of the dates and the future predictions in terms of price
         plt.plot(recursive_dates, recursive_predictions)