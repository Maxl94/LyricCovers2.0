import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_flow(
    df: pd.DataFrame,
    column_left: str,
    column_right: str,
    label_left: str,
    label_right: str,
    title: str = "",
) -> go.Figure:
    # Define a color map for languages
    classes = df[column_left].unique()

    # Use Plotly's built-in Pastel color scale for lighter colors
    color_scale = px.colors.qualitative.Pastel

    # Create a color map using the color scale
    color_map = {lang: color_scale[i % len(color_scale)] for i, lang in enumerate(classes)}

    # Define fixed order for languages
    class_order = classes

    # Create labels for both sides using the fixed order
    labels = [f"{class_item} ({label_left})" for class_item in class_order] + [
        f"{class_item} ({label_right})" for class_item in class_order
    ]

    # Create node colors using the fixed order
    node_colors = [color_map[lang] for lang in class_order] * 2

    # Create a mapping for source and target indices
    source_mapping = {lang: i for i, lang in enumerate(class_order)}
    target_mapping = {lang: i + len(class_order) for i, lang in enumerate(class_order)}

    # Create source and target lists using the mapping
    source_indices = [source_mapping[class_item] for class_item in df[column_left]]
    target_indices = [target_mapping[class_item] for class_item in df[column_right]]

    # Create link colors based on source language
    link_colors = [color_map[class_item] for class_item in df[column_left]]

    # Create node x-coordinates (0 for source, 1 for target)
    node_x = [0] * len(class_order) + [1] * len(class_order)

    # Create the Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    x=node_x,
                    color=node_colors,
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=df["count"],
                    color=link_colors,
                ),
            )
        ]
    )

    # Update the layout
    fig.update_layout(
        title=dict(text=title, x=0.5, y=0.95),
        font_size=12,
        height=600,
        width=1000,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig
