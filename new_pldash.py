import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

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
    },
    'chart_container': {
        'display': 'flex',
        'flexWrap': 'wrap',
        'gap': '20px'
    },
    'chart': {
        'flex': '1',
        'minWidth': '45%'
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
            html.Label("Security", style=styles['label']),
            dcc.Dropdown(
                id='security-dropdown',
                multi=True,
                style=styles['dropdown']
            )
        ]),
    ]),
    
    html.Div(style=styles['chart_container'], children=[
        dcc.Graph(id='revenue-chart', style=styles['chart']),
        dcc.Graph(id='percentage-chart', style=styles['chart'])
    ])
])

@app.callback(
    Output('security-dropdown', 'options'),
    Output('security-dropdown', 'value'),
    Input('market-dropdown', 'value')
)
def update_security_options(selected_market):
    if selected_market == 'All Markets':
        filtered_df = df
    else:
        filtered_df = df[df['market_type'] == selected_market]
    
    securities = filtered_df['security'].unique()
    options = [{'label': security, 'value': security} for security in sorted(securities)]
    return options, [options[0]['value']]  # Select the first security by default

@app.callback(
    [Output('revenue-chart', 'figure'),
     Output('percentage-chart', 'figure')],
    [Input('market-dropdown', 'value'),
     Input('security-dropdown', 'value')]
)
def update_charts(market, securities):
    if market != 'All Markets':
        filtered_df = df[df['market_type'] == market]
    else:
        filtered_df = df
    
    revenue_traces = []
    percentage_traces = []
    
    if securities:
        for security in securities:
            security_data = filtered_df[filtered_df['security'] == security]
            
            revenue_trace = go.Scatter(
                x=security_data['Date'],
                y=security_data['country_exposure_revenue'],
                mode='lines+markers',
                name=security,
                hovertemplate=
                "<b>%{text}</b><br>" +
                "Date: %{x}<br>" +
                "Revenue: %{y:.2f}<br>" +
                "<extra></extra>",
                text=security_data['security'],
                line=dict(width=2),
                marker=dict(size=6)
            )
            revenue_traces.append(revenue_trace)
            
            percentage_trace = go.Scatter(
                x=security_data['Date'],
                y=security_data['exposure_revenue_pct'],
                mode='lines+markers',
                name=security,
                hovertemplate=
                "<b>%{text}</b><br>" +
                "Date: %{x}<br>" +
                "Percentage: %{y:.2f}%<br>" +
                "<extra></extra>",
                text=security_data['security'],
                line=dict(width=2),
                marker=dict(size=6)
            )
            percentage_traces.append(percentage_trace)
    
    revenue_layout = go.Layout(
        title=f'Revenue Exposure - {market}',
        yaxis=dict(title='Revenue'),
        xaxis=dict(title='Dates'),
        legend=dict(orientation='h', y=1.1),
        showlegend=True,
        hovermode='closest'
    )
    
    percentage_layout = go.Layout(
        title=f'Percentage Revenue Exposure - {market}',
        yaxis=dict(title='Percentage of Revenue'),
        xaxis=dict(title='Dates'),
        legend=dict(orientation='h', y=1.1),
        showlegend=True,
        hovermode='closest'
    )
    
    return {'data': revenue_traces, 'layout': revenue_layout}, {'data': percentage_traces, 'layout': percentage_layout}

if __name__ == '__main__':
    app.run_server(debug=True, port=8000)