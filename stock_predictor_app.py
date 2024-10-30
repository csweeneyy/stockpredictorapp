'''
Connor Sweeney

'''


import streamlit as st 
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


start = "2015-01-01"
today = date.today().strftime("%Y-%m-%d")
st.title("Stock Prediction App")

stocks = ("AAPL", "NVDA", "GOOG", "VOO","VTI","MSFT")
selected_stock = st.selectbox("Select dataset for prediction", stocks)
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start, today)
    data.reset_index(inplace=True)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    return data


data_load_state = st.text("Loading data..")
data = load_data(selected_stock)
data_load_state.text("Loading data... Data Loaded!")

st.subheader("Raw Data")
st.write(data.tail())


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()


#forecasting
def prepare_prophet_data(data):
    # Make a copy and convert date column to timezone-naive
    df = data[['Date', 'Close']].copy()
    df['Date'] = df['Date'].dt.tz_localize(None)  # Remove timezone info
    df.columns = ['ds', 'y']  # Rename columns for Prophet
    return df

df_prophet = prepare_prophet_data(data)


m = Prophet()
m.fit(df_prophet)
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail())


st.write("Forecast Data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

#st.write("Made by CS")


