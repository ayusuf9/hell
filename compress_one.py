def compress_legend(fig, group1_base, group2_base, group1_color, group2_color, symbol_dict, big_df2):
    # Create two main legend entries for overweight and underweight
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color=group1_color),
        legendgroup=group1_base, name=group1_base, showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color=group2_color),
        legendgroup=group2_base, name=group2_base, showlegend=True
    ))

    # Process existing traces
    for i, trace in enumerate(fig.data[:-2]):  # Exclude the two traces we just added
        if 'name' in trace:
            part1, part2 = trace.name.split(',') if ',' in trace.name else (trace.name, '')
            if part1 in [group1_base, group2_base]:
                trace.legendgroup = part1
                trace.showlegend = False
                if part1 == group1_base:
                    trace.marker.color = group1_color
                    trace.line.color = group1_color if 'line' in trace else None
                else:
                    trace.marker.color = group2_color
                    trace.line.color = group2_color if 'line' in trace else None
            
            # Assign symbol legendgroup
            if 'marker' in trace and 'symbol' in trace.marker:
                symbol = trace.marker.symbol
                trace.legendgroup = f"symbol_{symbol}"
                trace.showlegend = False

    # Add symbol legend entries
    original_series_values = big_df2["Original series"].unique()
    for value in original_series_values:
        symbol = symbol_dict[value]
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(symbol=symbol, size=10, color='black'),
            name=value, showlegend=True,
            legendgroup=f"symbol_{symbol}",
            legendgrouptitle_text=value
        ))

    # Update layout for legend grouping
    fig.update_layout(
        legend=dict(
            groupclick="toggleitem"
        )
    )

    return fig