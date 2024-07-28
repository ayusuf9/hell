def compress_legend(fig, group1_base, group2_base, group1_color, group2_color, symbol_dict, big_df2):
    # Remove existing legend entries
    for trace in fig.data:
        trace.showlegend = False

    # Create toggleable legend entries for overweight and underweight
    for group, color in [(group1_base, group1_color), (group2_base, group2_color)]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color),
            legendgroup=group, name=group, showlegend=True,
            visible=True
        ))

    # Process existing traces and create symbol groups
    symbol_groups = {symbol: [] for symbol in symbol_dict.values()}
    for trace in fig.data[:-2]:  # Exclude the two traces we just added
        if 'marker' in trace:
            if trace.marker.color == group1_color:
                trace.legendgroup = group1_base
            elif trace.marker.color == group2_color:
                trace.legendgroup = group2_base
            
            if 'symbol' in trace.marker:
                symbol = trace.marker.symbol
                symbol_groups[symbol].append(trace)

    # Add symbol legend entries
    for value, symbol in symbol_dict.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(symbol=symbol, size=10, color='black'),
            name=value, showlegend=True,
            legendgroup=f"symbol_{symbol}",
            visible=True
        ))
        # Update all traces with this symbol to be part of this legend group
        for trace in symbol_groups[symbol]:
            trace.legendgroup += f"_symbol_{symbol}"

    # Update layout for legend grouping
    fig.update_layout(
        legend=dict(
            groupclick="toggleitem",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            itemsizing="constant",
            font=dict(size=18),
            orientation="h",
            x=1,
            y=1,
            tracegroupgap=5
        )
    )

    return fig