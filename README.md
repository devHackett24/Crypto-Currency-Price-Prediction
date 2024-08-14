Here's the updated README with the libraries you used:

---

# Cryptocurrency Price Prediction

## Project Overview
This project focuses on pulling real-time data from a cryptocurrency API and using historical price data to predict future cryptocurrency prices. The project leverages time series analysis, creating a model to recursively forecast future prices based on past prices.

## Technologies Used
- **Programming Languages:** Python
- **Libraries:**
  - `pandas`: For data manipulation and creating windowed DataFrames.
  - `numpy`: For numerical operations.
  - `tensorflow` and `keras`: For building and training the neural network model.
  - `matplotlib`: For visualizing the data and model predictions.
  - `scikit-learn (MinMaxScaler)`: For scaling the data before feeding it into the model.
  - `copy (deepcopy)`: To ensure immutability during data processing.
  - `datetime`: For handling date and time data.

## Key Features
- **Data Retrieval:**
  - Obtains `exchange_id`, `asset_id`, and `symbol_id` to retrieve historical data on cryptocurrency.
  
- **Data Preparation:**
  - Creation of a windowed DataFrame with a timestep of size three.
  - Conversion of the DataFrame into three-dimensional matrices suitable for time series modeling.
  - Data scaling using `MinMaxScaler` and splitting into training, validation, and testing sets.
  
- **Model Building:**
  - Utilizes the TensorFlow and Keras libraries to build a model incorporating LSTM layers for sequence prediction.
  - Includes hyperparameters such as Density, Dropout, and LSTM for optimized time series analysis.

- **Prediction:**
  - The trained model is used to recursively predict future cryptocurrency prices.


## Future Improvements
- **Model Optimization:** Experiment with different architectures and hyperparameters to further enhance prediction accuracy.
- **Additional Features:** Incorporate more cryptocurrencies and exchanges for broader analysis.
- **User Interface:** Develop a user-friendly interface for non-technical users to interact with the model.
