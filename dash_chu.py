import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
import random

# Data preparation
data = pd.read_csv('data.csv')
data['Date'] = pd.to_datetime(data['month_date']).dt.date
data['Fiscal_Date'] = pd.to_datetime(data['month_fiscal']).dt.to_period('M')
data['security_name'] = data['security_name'].astype('category')
data['iso_country_symbol'] = data['iso_country_symbol'].astype('category')
data['market_type'] = data['market_type'].astype('category')
data['sedol'] = data['sedol'].astype('category')

data['security'] = data['security_name'].astype(str) + "(" + data['sedol'].astype(str) + ")"
data['country_exposure_pct'] = data['country_exposure(pct)']

# Initializing the Dash app
app = dash.Dash(__name__)

df = data.copy()

# Get the first 3 unique securities
initial_securities = df['security'].unique()[:3]

# Define the layout
app.layout = html.Div([
    html.Div([
        html.H1("China Exposure Tool"),
        html.P("Second Phase of the demo", style={'color': 'white'}),
    ], style={'background-color': '#3366cc', 'color': 'white', 'padding': '20px'}),
    
    html.Div([
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=df['Date'].min(),
            end_date=df['Date'].max(),
            display_format='YYYY-MM-DD'
        ),
        dcc.Dropdown(
            id='market-dropdown',
            options=[{'label': 'All Markets', 'value': 'All Markets'}] + [{'label': market, 'value': market} for market in df['market_type'].unique()],
            value='All Markets',
            style={'width': '150px'}
        ),
        dcc.Dropdown(
            id='country-dropdown',
            options=[{'label': 'All', 'value': 'All'}] + [{'label': country, 'value': country} for country in df['iso_country_symbol'].unique()],
            value='All',
            style={'width': '150px'}
        ),
        dcc.Dropdown(
            id='security-dropdown',
            options=[{'label': security, 'value': security} for security in initial_securities],
            value=initial_securities.tolist(),
            multi=True,
            style={'width': '300px'}
        ),
        html.Button('Load More Securities', id='load-more-button', n_clicks=0),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px'}),
    
    dcc.Graph(id='security-chart'),
])

@app.callback(
    Output('security-dropdown', 'options'),
    Input('load-more-button', 'n_clicks'),
    State('security-dropdown', 'options')
)
def update_dropdown_options(n_clicks, current_options):
    if n_clicks == 0:
        return current_options
    
    all_securities = df['security'].unique()
    current_securities = set(option['value'] for option in current_options)
    
    # Choose 3 new random securities that are not already in the dropdown
    new_securities = random.sample([s for s in all_securities if s not in current_securities], 
                                   min(3, len(all_securities) - len(current_securities)))
    
    return current_options + [{'label': security, 'value': security} for security in new_securities]

@app.callback(
    Output('security-chart', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('market-dropdown', 'value'),
     Input('country-dropdown', 'value'),
     Input('security-dropdown', 'value')]
)
def update_chart(start_date, end_date, market, country, securities):
    # Convert string dates to datetime.date objects
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    if market != 'All Markets':
        filtered_df = filtered_df[filtered_df['market_type'] == market]
    if country != 'All':
        filtered_df = filtered_df[filtered_df['iso_country_symbol'] == country]
    
    traces = []
    if securities:  # Only create traces if securities are selected
        for security in securities:
            security_data = filtered_df[filtered_df['security'] == security]
            trace = go.Scatter(
                x=security_data['Date'],
                y=security_data['country_exposure_revenue'],
                mode='lines+markers',
                name=security,
                hovertemplate=
                "<b>%{customdata[0]}</b><br>" +
                "Date: %{x}<br>" +
                "Revenue: %{y:.2f}<br>" +
                "Exposure %: %{customdata[1]:.2f}%<br>" +
                "Country: %{customdata[2]}<br>" +
                "SEDOL: %{customdata[3]}<extra></extra>",
                customdata=security_data[['security', 'country_exposure_pct', 'iso_country_symbol', 'sedol']].values,
                line=dict(width=3),
                marker=dict(size=7) 
            )
            traces.append(trace)
    
    layout = go.Layout(
        title='Revenue Exposure',
        yaxis=dict(title='Revenue (Exposure)'),
        xaxis=dict(title='Dates'),
        legend=dict(orientation='h', y=1.1),
        showlegend=True,
        hovermode='closest'
    )
    
    return {'data': traces, 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True, port=8000)