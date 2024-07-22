import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
import random

# Generate sample data for multiple stocks
def generate_sample_data():
    end_date = datetime(2019, 11, 1)
    date_range = [end_date - timedelta(hours=i) for i in range(96, 0, -1)]
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']
    countries = ['France', 'Germany', 'Italy', 'Spain', 'United Kingdom']
    regions = {
        'France': 'Western Europe',
        'Germany': 'Western Europe',
        'Italy': 'Southern Europe',
        'Spain': 'Southern Europe',
        'United Kingdom': 'Northern Europe'
    }
    
    df = pd.DataFrame()
    for stock in stocks:
        stock_data = pd.DataFrame({
            'Date': date_range,
            'Price': [random.uniform(100, 200) + (random.random() - 0.5) * 10 for _ in range(96)],
            'Stock': stock,
            'Country': [random.choice(countries) for _ in range(96)]
        })
        stock_data['Region'] = stock_data['Country'].map(regions)
        df = pd.concat([df, stock_data])
    
    return df

# Initialize the Dash app
app = dash.Dash(__name__)

# Generate sample data
df = generate_sample_data()

# Define the layout
app.layout = html.Div([
    html.Div([
        html.H1("My Stocks Portfolio"),
        html.P("Currency in USD", style={'color': 'white'}),
    ], style={'background-color': '#3366cc', 'color': 'white', 'padding': '20px'}),
    
    html.Div([
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=df['Date'].min().date(),
            end_date=df['Date'].max().date(),
            display_format='MM/DD/YYYY'
        ),
        dcc.Dropdown(
            id='interval-dropdown',
            options=[
                {'label': '1H', 'value': '1H'},
                {'label': '4H', 'value': '4H'},
                {'label': '12H', 'value': '12H'},
                {'label': '1D', 'value': '1D'},
                {'label': '4D', 'value': '4D'},
                {'label': '1W', 'value': '1W'}
            ],
            value='1H',
            style={'width': '100px'}
        ),
        dcc.Dropdown(
            id='country-dropdown',
            options=[{'label': 'All', 'value': 'All'}] + [{'label': country, 'value': country} for country in df['Country'].unique()],
            value='All',
            style={'width': '150px'}
        ),
        dcc.Dropdown(
            id='region-dropdown',
            options=[{'label': 'All', 'value': 'All'}] + [{'label': region, 'value': region} for region in df['Region'].unique()],
            value='All',
            style={'width': '150px'}
        ),
        dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': stock, 'value': stock} for stock in df['Stock'].unique()],
            value=df['Stock'].unique().tolist(),
            multi=True,
            style={'width': '300px'}
        )
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px'}),
    
    dcc.Graph(id='stock-chart'),
])

@app.callback(
    Output('stock-chart', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('interval-dropdown', 'value'),
     Input('country-dropdown', 'value'),
     Input('region-dropdown', 'value'),
     Input('stock-dropdown', 'value')]
)
def update_chart(start_date, end_date, interval, country, region, stocks):
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    if country != 'All':
        filtered_df = filtered_df[filtered_df['Country'] == country]
    if region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == region]
    
    traces = []
    for stock in stocks:
        stock_data = filtered_df[filtered_df['Stock'] == stock]
        trace = go.Scatter(
            x=stock_data['Date'],
            y=stock_data['Price'],
            mode='lines',
            name=stock
        )
        traces.append(trace)
    
    layout = go.Layout(
        title='Stock Prices',
        yaxis=dict(title='Price (USD)'),
        legend=dict(orientation='h', y=1.1),
        showlegend=True
    )
    
    return {'data': traces, 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)