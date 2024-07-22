import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta

# Generate sample data
def generate_sample_data():
    end_date = datetime(2019, 11, 1)
    date_range = [end_date - timedelta(hours=i) for i in range(96, 0, -1)]
    prices = [1.43, 1.38, 1.65, 1.41, 1.52, 1.36, 1.32, 1.25, 1.2, 0.9, 0.98, 0.84, 0.81, 0.91, 0.86, 0.8, 0.85, 0.84, 0.82, 0.87, 0.86, 0.75, 0.72, 0.76, 0.74, 0.73]
    prices = prices + [prices[-1] + (random.random() - 0.5) * 0.1 for _ in range(70)]
    volumes = [random.randint(5, 25) / 100 for _ in range(96)]
    df = pd.DataFrame({
        'Date': date_range,
        'Price': prices,
        'Volume': volumes
    })
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
            id='chart-type-dropdown',
            options=[
                {'label': 'Line', 'value': 'line'},
                {'label': 'Candlestick', 'value': 'candlestick'}
            ],
            value='line',
            style={'width': '100px'}
        )
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px'}),
    
    dcc.Graph(id='stock-chart'),
    
    html.Div([
        html.Span("SNAP $13.65", style={'background-color': '#4CAF50', 'color': 'white', 'padding': '5px 10px', 'borderRadius': '5px'})
    ], style={'margin': '20px'})
])

@app.callback(
    Output('stock-chart', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('interval-dropdown', 'value'),
     Input('chart-type-dropdown', 'value')]
)
def update_chart(start_date, end_date, interval, chart_type):
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    if chart_type == 'line':
        trace = go.Scatter(x=filtered_df['Date'], y=filtered_df['Price'], mode='lines+markers')
    else:
        trace = go.Candlestick(x=filtered_df['Date'],
                               open=filtered_df['Price'],
                               high=filtered_df['Price'] + 0.1,
                               low=filtered_df['Price'] - 0.1,
                               close=filtered_df['Price'])
    
    volume_trace = go.Bar(x=filtered_df['Date'], y=filtered_df['Volume'], yaxis='y2', name='Volume')
    
    layout = go.Layout(
        title='Stock Price',
        yaxis=dict(title='Price (USD)'),
        yaxis2=dict(title='Volume', overlaying='y', side='right'),
        showlegend=False
    )
    
    return {'data': [trace, volume_trace], 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)