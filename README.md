Stock Prediction App
This app provides a user-friendly interface to forecast stock prices using historical data and machine learning. 
Built with the Streamlit library in python, the app allows users to choose from a selection of stocks and view future price predictions generated by the model.

Features
- Stock Selection: Choose from stocks like Apple (AAPL), Nvidia (NVDA), Google (GOOG), and more.
- Prediction Period: Set the prediction period for up to four years using a slider.
- Data Visualization: Displays raw historical stock data and a detailed forecast chart.
- Interactive Plotting: Includes dynamic plotting with Plotly for clear visual insights.
  
How to Use
- Select a Stock: Choose a stock from the dropdown menu.
- Set Prediction Period: Use the slider to set the prediction length (1-4 years).
- View Data & Forecast: The app displays recent stock data and future predictions, with interactive plotting.
  
Installation
To run the app locally:
1. Clone this repository.
2. Install the dependencies:
                                pip install streamlit yfinance prophet plotly
3. Run the application:
                                streamlit run stock_predictor_app.py

Technologies Used

Streamlit: For the web interface 

yfinance: To fetch historical stock data

Prophet: Machine learning model for time-series forecasting

Plotly: For interactive data visualization

This app offers a straightforward approach to stock prediction, useful for both investors and those interested in data science applications.
