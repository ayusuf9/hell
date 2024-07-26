import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

# Load data
data = pd.read_csv('data.csv')

# Preprocess data
data['Date'] = pd.to_datetime(data['month_date']).dt.date
data['Fiscal_Date'] = pd.to_datetime(data['month_fiscal']).dt.to_period('M')
data['security_name'] = data['security_name'].astype('category')
data['iso_country_symbol'] = data['iso_country_symbol'].astype('category')
data['market_type'] = data['market_type'].astype('category')
data['sedol'] = data['sedol'].astype('category')

data['security'] = data['security_name'].astype(str) + "(" + data['sedol'].astype(str) + ")"
data['country_exposure_pct'] = data['country_exposure(pct)']
data['country_exposure_revenue'] = data['country_exposure(revenue)']

# Create a list of unique market types
market_types = data['market_type'].unique()

# Create a Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Country Exposure Dashboard'),
    html.Div([
        html.Label('Market Type:'),
        dcc.Dropdown(
            id='market-type-dropdown',
            options=[{'label': market_type, 'value': market_type} for market_type in market_types],
            value=market_types[0]
        )
    ]),
    html.Div([
        html.Label('Securities:'),
        dcc.Dropdown(
            id='securities-dropdown',
            multi=True
        )
    ]),
    html.Div([
        html.Div([
            dcc.Graph(id='country-exposure-pct-graph')
        ], style={'width': '49%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='country-exposure-revenue-graph')
        ], style={'width': '49%', 'display': 'inline-block'})
    ])
])

# Define the callback to update the securities dropdown
@app.callback(
    Output('securities-dropdown', 'options'),
    [Input('market-type-dropdown', 'value')]
)
def update_securities_dropdown(market_type):
    securities = data[data['market_type'] == market_type]['security_name'].unique()
    return [{'label': security, 'value': security} for security in securities]

# Define the callback to update the graphs
@app.callback(
    [Output('country-exposure-pct-graph', 'figure'),
     Output('country-exposure-revenue-graph', 'figure')],
    [Input('market-type-dropdown', 'value'),
     Input('securities-dropdown', 'value')]
)
def update_graphs(market_type, securities):
    if securities is None:
        securities = []
    
    # Filter the data
    filtered_data = data[(data['market_type'] == market_type) & (data['security_name'].isin(securities))]
    
    # Create the figures
    fig_pct = go.Figure()
    fig_revenue = go.Figure()
    
    for i, security in enumerate(securities):
        security_data = filtered_data[filtered_data['security_name'] == security]
        fig_pct.add_trace(go.Scatter(
            x=security_data['Date'],
            y=security_data['country_exposure_pct'],
            name=security,
            mode='lines',
            marker=dict(symbol=i)
        ))
        fig_revenue.add_trace(go.Scatter(
            x=security_data['Date'],
            y=security_data['country_exposure_revenue'],
            name=security,
            mode='lines',
            marker=dict(symbol=i)
        ))
    
    # Add horizontal lines
    max_value_pct = filtered_data['country_exposure_pct'].max()
    min_value_pct = filtered_data['country_exposure_pct'].min()
    fig_pct.add_trace(go.Scatter(
        x=[filtered_data['Date'].min(), filtered_data['Date'].max()],
        y=[max_value_pct + 0.5] * 2,
        mode='lines',
        line=dict(color='grey', width=3),
        showlegend=False
    ))
    fig_pct.add_trace(go.Scatter(
        x=[filtered_data['Date'].min(), filtered_data['Date'].max()],
        y=[min_value_pct - 0.5] * 2,
        mode='lines',
        line=dict(color='grey', width=3),
        showlegend=False
    ))
    
    max_value_revenue = filtered_data['country_exposure_revenue'].max()
    min_value_revenue = filtered_data['country_exposure_revenue'].min()
    fig_revenue.add_trace(go.Scatter(
        x=[filtered_data['Date'].min(), filtered_data['Date'].max()],
        y=[max_value_revenue + 0.5] * 2,
        mode='lines',
        line=dict(color='grey', width=3),
        showlegend=False
    ))
    fig_revenue.add_trace(go.Scatter(
        x=[filtered_data['Date'].min(), filtered_data['Date'].max()],
        y=[min_value_revenue - 0.5] * 2,
        mode='lines',
        line=dict(color='grey', width=3),
        showlegend=False
    ))
    
    # Update the layout
    fig_pct.update_layout(
        title_text="Country Exposure (%)",
        xaxis_title="Date",
        yaxis_title="Country Exposure (%)",
        hovermode="x unified",
        template="plotly_white",
        width=800,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_revenue.update_layout(
        title_text="Country Exposure (Revenue)",
        xaxis_title="Date",
        yaxis_title="Country Exposure (Revenue)",
        hovermode="x unified",
        template="plotly_white",
        width=800,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig_pct, fig_revenue

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)