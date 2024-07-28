import plotly.graph_objects as go

def compress_legend(fig, group1_base, group2_base, group1_color, group2_color, symbol_dict, big_df2):
    # Create dictionaries to store traces by group and symbol
    group_traces = {group1_base: [], group2_base: []}
    symbol_traces = {symbol: [] for symbol in symbol_dict.keys()}
    
    # Iterate through existing traces and categorize them
    for trace in fig.data:
        if trace.name.startswith(group1_base) or trace.name.startswith(group2_base):
            group = group1_base if trace.name.startswith(group1_base) else group2_base
            symbol = trace.marker.symbol
            group_traces[group].append(trace)
            symbol_traces[symbol].append(trace)
            
            # Update trace properties
            trace.name = f"{group} - {symbol}"
            trace.legendgroup = f"{group} - {symbol}"
            trace.showlegend = False
    
    # Create legend entries for color groups (overweight/underweight)
    for group, color in [(group1_base, group1_color), (group2_base, group2_color)]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            name=group,
            mode='markers',
            marker=dict(color=color, size=10),
            legendgroup=group,
            showlegend=True
        ))
    
    # Create legend entries for symbols
    for symbol, marker_symbol in symbol_dict.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            name=symbol,
            mode='markers',
            marker=dict(symbol=marker_symbol, size=10, color='black'),
            legendgroup=symbol,
            showlegend=True
        ))
    
    # Custom click handler for legend items
    def update_traces(trace, clickData):
        # Determine if it's a group (color) or symbol click
        is_group_click = trace.name in [group1_base, group2_base]
        
        # Handle double-click (isolation)
        if clickData.doubleclick:
            for t in fig.data:
                t.visible = False
            if is_group_click:
                for t in group_traces[trace.name]:
                    t.visible = True
            else:
                for t in symbol_traces[trace.name]:
                    t.visible = True
            trace.visible = True
        else:
            # Handle single-click (toggle)
            new_visible = not trace.visible
            if is_group_click:
                for t in group_traces[trace.name]:
                    t.visible = new_visible
            else:
                for t in symbol_traces[trace.name]:
                    t.visible = new_visible
            trace.visible = new_visible
        
        return fig
    
    # Apply the custom click handler to all legend items
    for trace in fig.data:
        if trace.showlegend:
            trace.on_click = update_traces
    
    return fig