import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_sample_data, create_gauge_chart

# Page configuration
st.set_page_config(
    page_title="Data Visualization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visuals
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
        padding: 20px;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1E88E5, #64B5F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Subheader styling */
    .sub-header {
        font-size: 1.5rem;
        color: #1976D2;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Card styling */
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1976D2;
        color: white;
        font-weight: 500;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1565C0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Info box styling */
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #1976D2;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        color: #757575;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Main content area
st.markdown("<h1 class='main-header'>Interactive Data Visualization</h1>", unsafe_allow_html=True)

# Create sample data
if 'data' not in st.session_state:
    st.session_state.data = create_sample_data(1000)

df = st.session_state.data

# Sidebar with visualization options
with st.sidebar:
    st.markdown("### Visualization Options")
    
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Heatmap", "Gauge Chart", "Funnel Chart"]
    )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Common options for all chart types
    if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
        x_axis = st.selectbox(
            "X-Axis",
            ["category", "region", "date"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        y_axis = st.selectbox(
            "Y-Axis",
            ["sales", "units", "avg_price"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        color_by = st.selectbox(
            "Color By",
            ["None", "category", "region"],
            format_func=lambda x: x.replace("_", " ").title() if x != "None" else x
        )
    
    # Chart-specific options
    if chart_type == "Bar Chart":
        orientation = st.radio("Orientation", ["Vertical", "Horizontal"])
        
    elif chart_type == "Line Chart":
        line_shape = st.selectbox("Line Shape", ["linear", "spline", "hv", "vh", "hvh", "vhv"])
        markers = st.checkbox("Show Markers", value=True)
        
    elif chart_type == "Scatter Plot":
        size_by = st.selectbox(
            "Size By",
            ["None", "sales", "units", "avg_price"],
            format_func=lambda x: x.replace("_", " ").title() if x != "None" else x
        )
        
    elif chart_type == "Pie Chart":
        values = st.selectbox(
            "Values",
            ["sales", "units", "avg_price"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        names = st.selectbox(
            "Names",
            ["category", "region"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        donut = st.checkbox("Donut Chart", value=False)
        
    elif chart_type == "Heatmap":
        x_var = st.selectbox(
            "X-Axis",
            ["category", "region"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        y_var = st.selectbox(
            "Y-Axis",
            ["region", "category"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        z_var = st.selectbox(
            "Values",
            ["sales", "units", "avg_price"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
    elif chart_type == "Gauge Chart":
        metric = st.selectbox(
            "Metric",
            ["sales", "units", "avg_price"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        aggregation = st.selectbox(
            "Aggregation",
            ["Mean", "Median", "Sum", "Min", "Max"]
        )
        
    elif chart_type == "Funnel Chart":
        stages = st.multiselect(
            "Stages",
            ["Impressions", "Clicks", "Add to Cart", "Checkout", "Purchase"],
            default=["Impressions", "Clicks", "Add to Cart", "Checkout", "Purchase"]
        )

# Main content based on selected chart type
st.markdown("<div class='card'>", unsafe_allow_html=True)

if chart_type == "Bar Chart":
    st.markdown(f"<h3 class='sub-header'>Bar Chart: {y_axis.replace('_', ' ').title()} by {x_axis.replace('_', ' ').title()}</h3>", unsafe_allow_html=True)
    
    # Prepare data
    if x_axis == "date":
        # Group by month for date axis
        chart_df = df.groupby(pd.Grouper(key="date", freq="M"))[y_axis].mean().reset_index()
        chart_df["date"] = chart_df["date"].dt.strftime("%b %Y")
    else:
        # Group by category or region
        if color_by != "None" and color_by != x_axis:
            chart_df = df.groupby([x_axis, color_by])[y_axis].mean().reset_index()
        else:
            chart_df = df.groupby(x_axis)[y_axis].mean().reset_index()
    
    # Create bar chart
    if orientation == "Vertical":
        if color_by != "None" and color_by != x_axis:
            fig = px.bar(
                chart_df,
                x=x_axis,
                y=y_axis,
                color=color_by,
                barmode="group",
                template="plotly_white"
            )
        else:
            fig = px.bar(
                chart_df,
                x=x_axis,
                y=y_axis,
                color=x_axis,
                template="plotly_white"
            )
    else:  # Horizontal
        if color_by != "None" and color_by != x_axis:
            fig = px.bar(
                chart_df,
                y=x_axis,
                x=y_axis,
                color=color_by,
                barmode="group",
                orientation="h",
                template="plotly_white"
            )
        else:
            fig = px.bar(
                chart_df,
                y=x_axis,
                x=y_axis,
                color=x_axis,
                orientation="h",
                template="plotly_white"
            )
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title=x_axis.replace("_", " ").title() if orientation == "Vertical" else y_axis.replace("_", " ").title(),
        yaxis_title=y_axis.replace("_", " ").title() if orientation == "Vertical" else x_axis.replace("_", " ").title()
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Line Chart":
    st.markdown(f"<h3 class='sub-header'>Line Chart: {y_axis.replace('_', ' ').title()} over {x_axis.replace('_', ' ').title()}</h3>", unsafe_allow_html=True)
    
    # Prepare data
    if x_axis == "date":
        # Group by month for date axis
        if color_by != "None":
            chart_df = df.groupby([pd.Grouper(key="date", freq="M"), color_by])[y_axis].mean().reset_index()
        else:
            chart_df = df.groupby(pd.Grouper(key="date", freq="M"))[y_axis].mean().reset_index()
    else:
        # Group by category or region
        if color_by != "None" and color_by != x_axis:
            chart_df = df.groupby([x_axis, color_by])[y_axis].mean().reset_index()
        else:
            chart_df = df.groupby(x_axis)[y_axis].mean().reset_index()
    
    # Create line chart
    if color_by != "None":
        fig = px.line(
            chart_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            markers=markers,
            line_shape=line_shape,
            template="plotly_white"
        )
    else:
        fig = px.line(
            chart_df,
            x=x_axis,
            y=y_axis,
            markers=markers,
            line_shape=line_shape,
            template="plotly_white"
        )
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title=x_axis.replace("_", " ").title(),
        yaxis_title=y_axis.replace("_", " ").title()
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Scatter Plot":
    st.markdown(f"<h3 class='sub-header'>Scatter Plot: {y_axis.replace('_', ' ').title()} vs {x_axis.replace('_', ' ').title()}</h3>", unsafe_allow_html=True)
    
    # Create scatter plot
    if color_by != "None":
        if size_by != "None":
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color=color_by,
                size=size_by,
                template="plotly_white",
                opacity=0.7
            )
        else:
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color=color_by,
                template="plotly_white",
                opacity=0.7
            )
    else:
        if size_by != "None":
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                size=size_by,
                template="plotly_white",
                opacity=0.7
            )
        else:
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                template="plotly_white",
                opacity=0.7
            )
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title=x_axis.replace("_", " ").title(),
        yaxis_title=y_axis.replace("_", " ").title()
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Pie Chart":
    st.markdown(f"<h3 class='sub-header'>Pie Chart: {values.replace('_', ' ').title()} by {names.replace('_', ' ').title()}</h3>", unsafe_allow_html=True)
    
    # Prepare data
    chart_df = df.groupby(names)[values].sum().reset_index()
    
    # Create pie chart
    fig = px.pie(
        chart_df,
        values=values,
        names=names,
        hole=0.4 if donut else 0,
        template="plotly_white"
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add table with values
    st.markdown("<h4>Data Table</h4>", unsafe_allow_html=True)
    
    # Calculate percentages
    chart_df["Percentage"] = (chart_df[values] / chart_df[values].sum() * 100).round(2)
    chart_df["Percentage"] = chart_df["Percentage"].astype(str) + "%"
    
    # Rename columns for display
    display_df = chart_df.copy()
    display_df.columns = [col.replace("_", " ").title() for col in display_df.columns]
    
    st.dataframe(display_df, use_container_width=True)

elif chart_type == "Heatmap":
    st.markdown(f"<h3 class='sub-header'>Heatmap: {z_var.replace('_', ' ').title()} by {x_var.replace('_', ' ').title()} and {y_var.replace('_', ' ').title()}</h3>", unsafe_allow_html=True)
    
    # Prepare data
    pivot_df = df.pivot_table(
        index=y_var,
        columns=x_var,
        values=z_var,
        aggfunc="mean"
    )
    
    # Create heatmap
    fig = px.imshow(
        pivot_df,
        color_continuous_scale="Blues",
        template="plotly_white",
        labels=dict(color=z_var.replace("_", " ").title())
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title=x_var.replace("_", " ").title(),
        yaxis_title=y_var.replace("_", " ").title()
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add annotations to the heatmap
    st.markdown("<h4>Heatmap with Annotations</h4>", unsafe_allow_html=True)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale="Blues",
        hoverongaps=False
    ))
    
    # Add annotations
    annotations = []
    for i, row in enumerate(pivot_df.index):
        for j, col in enumerate(pivot_df.columns):
            annotations.append(
                dict(
                    x=col,
                    y=row,
                    text=str(round(pivot_df.iloc[i, j], 1)),
                    showarrow=False,
                    font=dict(color="white" if pivot_df.iloc[i, j] > pivot_df.values.mean() else "black")
                )
            )
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title=x_var.replace("_", " ").title(),
        yaxis_title=y_var.replace("_", " ").title(),
        annotations=annotations
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Gauge Chart":
    st.markdown(f"<h3 class='sub-header'>Gauge Chart: {metric.replace('_', ' ').title()} ({aggregation})</h3>", unsafe_allow_html=True)
    
    # Calculate the value based on aggregation
    if aggregation == "Mean":
        value = df[metric].mean()
    elif aggregation == "Median":
        value = df[metric].median()
    elif aggregation == "Sum":
        value = df[metric].sum()
    elif aggregation == "Min":
        value = df[metric].min()
    else:  # Max
        value = df[metric].max()
    
    # Create gauge chart
    fig = create_gauge_chart(
        value=value,
        min_val=0,
        max_val=df[metric].max() * 1.2,
        title=f"{metric.replace('_', ' ').title()} ({aggregation})"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add breakdown by category and region
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>By Category</h4>", unsafe_allow_html=True)
        
        # Calculate values by category
        if aggregation == "Mean":
            cat_values = df.groupby("category")[metric].mean()
        elif aggregation == "Median":
            cat_values = df.groupby("category")[metric].median()
        elif aggregation == "Sum":
            cat_values = df.groupby("category")[metric].sum()
        elif aggregation == "Min":
            cat_values = df.groupby("category")[metric].min()
        else:  # Max
            cat_values = df.groupby("category")[metric].max()
        
        # Create mini gauge charts for each category
        for cat, val in cat_values.items():
            st.markdown(f"<h5>{cat}</h5>", unsafe_allow_html=True)
            
            fig = create_gauge_chart(
                value=val,
                min_val=0,
                max_val=df[metric].max() * 1.2,
                title=""
            )
            
            fig.update_layout(height=150, margin=dict(l=20, r=20, t=0, b=0))
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<h4>By Region</h4>", unsafe_allow_html=True)
        
        # Calculate values by region
        if aggregation == "Mean":
            reg_values = df.groupby("region")[metric].mean()
        elif aggregation == "Median":
            reg_values = df.groupby("region")[metric].median()
        elif aggregation == "Sum":
            reg_values = df.groupby("region")[metric].sum()
        elif aggregation == "Min":
            reg_values = df.groupby("region")[metric].min()
        else:  # Max
            reg_values = df.groupby("region")[metric].max()
        
        # Create mini gauge charts for each region
        for reg, val in reg_values.items():
            st.markdown(f"<h5>{reg}</h5>", unsafe_allow_html=True)
            
            fig = create_gauge_chart(
                value=val,
                min_val=0,
                max_val=df[metric].max() * 1.2,
                title=""
            )
            
            fig.update_layout(height=150, margin=dict(l=20, r=20, t=0, b=0))
            
            st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Funnel Chart":
    st.markdown("<h3 class='sub-header'>Funnel Chart: Conversion Funnel</h3>", unsafe_allow_html=True)
    
    # Create sample funnel data
    np.random.seed(42)
    
    # Start with a base number and apply decreasing conversion rates
    base = 10000
    conversion_rates = [1.0, 0.4, 0.2, 0.1, 0.05]
    
    values = [int(base * rate * (1 + np.random.uniform(-0.1, 0.1))) for rate in conversion_rates[:len(stages)]]
    
    # Create funnel chart
    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textinfo="value+percent initial",
        marker={"color": px.colors.sequential.Blues[::-1][:len(stages)]}
    ))
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add conversion rates table
    st.markdown("<h4>Conversion Rates</h4>", unsafe_allow_html=True)
    
    # Calculate step-by-step conversion rates
    step_rates = []
    
    for i in range(len(stages) - 1):
        rate = values[i+1] / values[i] * 100
        step_rates.append({
            "From": stages[i],
            "To": stages[i+1],
            "Conversion Rate": f"{rate:.2f}%",
            "Absolute": f"{values[i+1]} / {values[i]}"
        })
    
    # Create DataFrame for display
    rates_df = pd.DataFrame(step_rates)
    
    st.dataframe(rates_df, use_container_width=True)
    
    # Add visualization of conversion rates
    st.markdown("<h4>Step-by-Step Conversion Visualization</h4>", unsafe_allow_html=True)
    
    # Create horizontal bar chart for conversion rates
    conversion_df = pd.DataFrame({
        "Step": [f"{step_rates[i]['From']} → {step_rates[i]['To']}" for i in range(len(step_rates))],
        "Rate": [float(step_rates[i]['Conversion Rate'].replace('%', '')) for i in range(len(step_rates))]
    })
    
    fig = px.bar(
        conversion_df,
        y="Step",
        x="Rate",
        orientation="h",
        color="Rate",
        color_continuous_scale="Blues",
        template="plotly_white"
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Conversion Rate (%)",
        yaxis_title="",
        coloraxis_showscale=False
    )
    
    # Add target lines
    fig.add_shape(
        type="line",
        x0=20,
        y0=-0.5,
        x1=20,
        y1=len(step_rates) - 0.5,
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=20,
        y=len(step_rates) - 0.5,
        text="Target: 20%",
        showarrow=False,
        yshift=10,
        font=dict(color="red")
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# Add a footer
st.markdown("""
<div class="footer">
    <p>© 2023 Your Company Name. All rights reserved.</p>
    <p>Created with Streamlit • Last updated: December 2023</p>
</div>
""", unsafe_allow_html=True)

