import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Data for each platform
data = {
    'Facebook': [62, 64, 64, 66, 78],
    'Instagram': [16, 28, 34, 42, 53],
    'LinkedIn': [20, 22, 27, 22, 29],
    'Twitter': [18, 19, 21, 21, 24]
}
years = [2012, 2013, 2014, 2015, 2016]

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Reach of Social Media Platforms Among Youth (2012-2016)"),
    dcc.Checklist(
        id='platform-checklist',
        options=[{'label': platform, 'value': platform} for platform in data.keys()],
        value=['Facebook'],  # Default selected platform
        inline=True
    ),
    dcc.Graph(id='social-media-graph')
])

# Callback to update the graph based on checklist selection
@app.callback(
    Output('social-media-graph', 'figure'),
    Input('platform-checklist', 'value')
)
def update_graph(selected_platforms):
    fig = go.Figure()

    for platform in selected_platforms:
        fig.add_trace(go.Scatter(
            x=years,
            y=data[platform],
            mode='lines+markers',
            name=platform
        ))

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='% of youth on this platform',
        yaxis_range=[0, 80],
        legend_title='Platform',
        font=dict(family="Arial", size=12),
        height=600,
        width=800
    )

    fig.update_yaxes(ticksuffix='%', tick0=0, dtick=20)

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)