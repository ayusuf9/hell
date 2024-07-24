import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta

# Data preparation
data = pd.read_csv('data.csv')
data['Date'] = pd.to_datetime(data['month_date']).dt.date
data['Fiscal_Date'] = pd.to_datetime(data['month_fiscal']).dt.to_period('M')
data['security_name'] = data['security_name'].astype('category')
data['iso_country_symbol'] = data['iso_country_symbol'].astype('category')
data['market_type'] = data['market_type'].astype('category')
data['sedol'] = data['sedol'].astype('category')

data['security'] = data['security_name'].astype(str) + " (" + data['sedol'].astype(str) + ")"
data['country_exposure_pct'] = data['country_exposure(pct)']

# Initializing the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

df = data.copy()

# Define styles
styles = {
    'container': {
        'margin': '20px',
        'fontFamily': 'Arial, sans-serif'
    },
    'header': {
        'backgroundColor': '#3366cc',
        'color': 'white',
        'padding': '20px',
        'marginBottom': '20px'
    },
    'filter_container': {
        'display': 'flex',
        'flexWrap': 'wrap',
        'gap': '20px',
        'marginBottom': '20px',
        'zIndex': 1000,
        'position': 'relative'
    },
    'filter_item': {
        'flex': '1',
        'minWidth': '200px',
        'zIndex': 1000
    },
    'label': {
        'marginBottom': '5px',
        'fontWeight': 'bold'
    },
    'dropdown': {
        'zIndex': 1001
    }
}

# Define the layout
app.layout = html.Div(style=styles['container'], children=[
    html.Div(style=styles['header'], children=[
        html.H1("China Exposure Tool"),
        html.P("Enhanced Filtering Demo")
    ]),
    
    html.Div(style=styles['filter_container'], children=[
        html.Div(style=styles['filter_item'], children=[
            html.Label("Date Range", style=styles['label']),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=df['Date'].min(),
                end_date=df['Date'].max(),
                display_format='YYYY-MM-DD',
                style={'zIndex': 1001}
            )
        ]),
        html.Div(style=styles['filter_item'], children=[
            html.Label("Market", style=styles['label']),
            dcc.Dropdown(
                id='market-dropdown',
                options=[{'label': 'All Markets', 'value': 'All Markets'}] + 
                        [{'label': market.capitalize(), 'value': market} for market in df['market_type'].unique()],
                value='All Markets',
                style=styles['dropdown']
            )
        ]),
        html.Div(style=styles['filter_item'], children=[
            html.Label("Country", style=styles['label']),
            dcc.Dropdown(
                id='country-dropdown',
                style=styles['dropdown']
            )
        ]),
        html.Div(style=styles['filter_item'], children=[
            html.Label("Security", style=styles['label']),
            dcc.Dropdown(
                id='security-dropdown',
                multi=True,
                style=styles['dropdown']
            )
        ]),
    ]),
    
    dcc.Graph(id='security-chart')
])

@app.callback(
    Output('country-dropdown', 'options'),
    Output('country-dropdown', 'value'),
    Input('market-dropdown', 'value')
)
def update_country_options(selected_market):
    if selected_market == 'All Markets':
        countries = df['iso_country_symbol'].unique()
    else:
        countries = df[df['market_type'] == selected_market]['iso_country_symbol'].unique()
    
    options = [{'label': country, 'value': country} for country in sorted(countries)]
    return options, options[0]['value']  # Select the first country by default

@app.callback(
    Output('security-dropdown', 'options'),
    Output('security-dropdown', 'value'),
    Input('market-dropdown', 'value'),
    Input('country-dropdown', 'value')
)
def update_security_options(selected_market, selected_country):
    if selected_market == 'All Markets':
        filtered_df = df[df['iso_country_symbol'] == selected_country]
    else:
        filtered_df = df[(df['market_type'] == selected_market) & (df['iso_country_symbol'] == selected_country)]
    
    securities = filtered_df['security'].unique()
    options = [{'label': security, 'value': security} for security in sorted(securities)]
    return options, [options[0]['value']]  # Select the first security by default

@app.callback(
    Output('security-chart', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('market-dropdown', 'value'),
     Input('country-dropdown', 'value'),
     Input('security-dropdown', 'value')]
)
def update_chart(start_date, end_date, market, country, securities):
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    if market != 'All Markets':
        filtered_df = filtered_df[filtered_df['market_type'] == market]
    
    filtered_df = filtered_df[filtered_df['iso_country_symbol'] == country]
    
    traces = []
    if securities:
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
        title=f'Revenue Exposure - {country}',
        yaxis=dict(title='Revenue (Exposure)'),
        xaxis=dict(title='Dates'),
        legend=dict(orientation='h', y=1.1),
        showlegend=True,
        hovermode='closest'
    )
    
    return {'data': traces, 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True, port=8000)