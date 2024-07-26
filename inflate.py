import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

# Assume data preparation code is already run as provided

app = dash.Dash(__name__)

# Define exact color scheme
colors = {
    'CPI': 'rgb(0, 0, 255)',
    'CPI_Core': 'rgb(255, 0, 0)',
    'Non_food': 'rgb(128, 128, 128)',
    'Food': 'rgb(173, 216, 230)',
    'US': 'rgb(0, 0, 255)',
    'Japan': 'rgb(255, 0, 0)',
    'UK': 'rgb(128, 128, 128)',
    'Germany': 'rgb(173, 216, 230)',
    'China_World': 'rgb(255, 0, 255)'
}

def create_figure(market, securities):
    filtered_df = data[data['market_type'] == market] if market != 'All Markets' else data
    
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, horizontal_spacing=0.02,
                        subplot_titles=("China: Consumer price inflation, y/y %", "Consumer price inflation around the world"),
                        specs=[[{"secondary_y": True}, {"secondary_y": False}]])

    # Left chart (China)
    for i, security in enumerate(securities[:4]):  # Limit to 4 securities for left chart
        security_data = filtered_df[filtered_df['security'] == security]
        
        fig.add_trace(go.Scatter(
            x=security_data['Date'],
            y=security_data['country_exposure_revenue'],
            name=security,
            line=dict(color=list(colors.values())[i], width=2),
            legendgroup=f"group{i}",
            showlegend=True
        ), row=1, col=1, secondary_y=False)

    # Add Food (RHS) trace with dashed line
    if len(securities) > 3:
        security_data = filtered_df[filtered_df['security'] == securities[3]]
        fig.add_trace(go.Scatter(
            x=security_data['Date'],
            y=security_data['country_exposure_revenue'],
            name=f"{securities[3]} (RHS)",
            line=dict(color=colors['Food'], width=2, dash='dot'),
            legendgroup="group3",
            showlegend=True
        ), row=1, col=1, secondary_y=True)

    # Right chart (World)
    for i, security in enumerate(securities):
        security_data = filtered_df[filtered_df['security'] == security]
        
        fig.add_trace(go.Scatter(
            x=security_data['Date'],
            y=security_data['country_exposure_revenue'],
            name=security,
            line=dict(color=list(colors.values())[i], width=2),
            legendgroup=f"group{i}",
            showlegend=False
        ), row=1, col=2)

    # Update layout
    fig.update_layout(
        height=600, width=1200,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        font=dict(family="Arial", size=10),
        margin=dict(l=50, r=50, t=100, b=50),
        plot_bgcolor='white',
    )

    # Update axes
    fig.update_xaxes(showgrid=True, gridcolor='lightgrey', showline=True, linecolor='black', mirror=True,
                     rangeslider_visible=False, tickformat="%Y")
    fig.update_yaxes(showgrid=True, gridcolor='lightgrey', showline=True, linecolor='black', mirror=True,
                     tickformat=".1f")

    # Specific adjustments for left chart
    fig.update_yaxes(title_text="", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="", secondary_y=True, row=1, col=1, overlaying="y")

    # Specific adjustments for right chart
    fig.update_yaxes(title_text="", row=1, col=2)
    
    # Add vertical line for 2015 in the right chart
    fig.add_vline(x="2015-01-01", line_width=1, line_dash="dash", line_color="black", col=2)

    # Add annotations for 2015 values
    annotations = [
        dict(x="2015-12-01", y=0.73, xref="x2", yref="y2", text="US : (Dec 2015) 0.73", showarrow=False, font=dict(size=8), xanchor="left", yanchor="bottom"),
        dict(x="2015-12-01", y=0.20, xref="x2", yref="y2", text="Japan : (Dec 2015) 0.20", showarrow=False, font=dict(size=8), xanchor="left", yanchor="bottom"),
        dict(x="2015-12-01", y=0.50, xref="x2", yref="y2", text="UK : (Dec 2015) 0.50", showarrow=False, font=dict(size=8), xanchor="left", yanchor="bottom"),
        dict(x="2015-12-01", y=0.47, xref="x2", yref="y2", text="Germany : (Dec 2015) 0.47", showarrow=False, font=dict(size=8), xanchor="left", yanchor="bottom"),
        dict(x="2015-12-01", y=1.60, xref="x2", yref="y2", text="China : (Dec 2015) 1.60", showarrow=False, font=dict(size=8), xanchor="left", yanchor="bottom")
    ]
    fig.update_layout(annotations=annotations)

    # Add date range buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.1,
                y=1.1,
                buttons=list([
                    dict(label="1YR", method="relayout", args=[{"xaxis.range": [data['Date'].max() - pd.DateOffset(years=1), data['Date'].max()]}]),
                    dict(label="5YR", method="relayout", args=[{"xaxis.range": [data['Date'].max() - pd.DateOffset(years=5), data['Date'].max()]}]),
                    dict(label="10YR", method="relayout", args=[{"xaxis.range": [data['Date'].max() - pd.DateOffset(years=10), data['Date'].max()]}]),
                    dict(label="Max", method="relayout", args=[{"xaxis.range": [data['Date'].min(), data['Date'].max()]}]),
                ]),
            )
        ]
    )

    # Add data source information
    current_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    fig.add_annotation(x=1, y=-0.15, xref='paper', yref='paper', text=f'Data as of: {current_date}', showarrow=False, font=dict(size=10), xanchor='right')

    return fig

app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='market-dropdown',
            options=[{'label': 'All Markets', 'value': 'All Markets'}] + 
                    [{'label': market, 'value': market} for market in data['market_type'].unique()],
            value='All Markets'
        ),
        dcc.Dropdown(
            id='security-dropdown',
            multi=True
        )
    ], style={'width': '50%', 'display': 'inline-block'}),
    dcc.Graph(id='inflation-charts')
])

@app.callback(
    Output('security-dropdown', 'options'),
    Output('security-dropdown', 'value'),
    Input('market-dropdown', 'value')
)
def update_security_options(selected_market):
    if selected_market == 'All Markets':
        filtered_df = data
    else:
        filtered_df = data[data['market_type'] == selected_market]
    
    securities = filtered_df['security'].unique()
    options = [{'label': security, 'value': security} for security in securities]
    return options, options[:5]  # Select first 5 securities by default

@app.callback(
    Output('inflation-charts', 'figure'),
    Input('market-dropdown', 'value'),
    Input('security-dropdown', 'value')
)
def update_charts(market, securities):
    return create_figure(market, securities)

if __name__ == '__main__':
    app.run_server(debug=True)