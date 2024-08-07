def compress_legend(fig, group1_base, group2_base, group1_color, group2_color, symbol_dict, big_df2):
    # Remove existing legend entries
    for trace in fig.data:
        trace.showlegend = False

    # Create toggleable legend entries for overweight and underweight
    for group, color in [(group1_base, group1_color), (group2_base, group2_color)]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color),
            name=group, showlegend=True,
            legendgroup=group
        ))

    # Process existing traces
    for trace in fig.data[:-2]:  # Exclude the two traces we just added
        if 'marker' in trace:
            if trace.marker.color == group1_color:
                trace.legendgroup = group1_base
            elif trace.marker.color == group2_color:
                trace.legendgroup = group2_base

    # Add symbol legend entries
    for value, symbol in symbol_dict.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(symbol=symbol, size=10, color='black'),
            name=value, showlegend=True,
            legendgroup=f"symbol_{value}"
        ))

    # Update layout for legend grouping
    fig.update_layout(
        legend=dict(
            groupclick="toggleitem",
            itemsizing="constant",
            font=dict(size=18),
            orientation="h",
            x=1,
            y=1,
        )
    )

    return fig