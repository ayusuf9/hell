import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
import random

# ... (previous code remains the same)

app.layout = html.Div([
    # ... (previous layout code remains the same)
    
    dcc.Graph(id='stock-chart'),
    
    html.Div(id='tooltip', style={'display': 'none'}, className='tooltip')
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
    # ... (previous update_chart function code remains the same)
    return {'data': traces, 'layout': layout}

app.clientside_callback(
    """
    function(hoverData, clickData, figure) {
        if(hoverData) {
            var point = hoverData.points[0];
            var data = point.customdata;
            var content = `
                <strong>${data.Stock}</strong><br>
                Price: $${data.Price.toFixed(2)}<br>
                Country: ${data.Country}<br>
                Region: ${data.Region}<br>
                ${data.CountryInfo}
            `;
            return [content, point.x + 10, point.y + 10, 'block'];
        }
        return ['', 0, 0, 'none'];
    }
    """,
    Output('tooltip', 'children'),
    Output('tooltip', 'style.left'),
    Output('tooltip', 'style.top'),
    Output('tooltip', 'style.display'),
    Input('stock-chart', 'hoverData'),
    Input('stock-chart', 'clickData'),
    State('stock-chart', 'figure')
)

if __name__ == '__main__':
    app.run_server(debug=True)