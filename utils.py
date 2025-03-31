import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64

def create_sample_data(n_rows=1000):
    """
    Create sample data for demonstration purposes
    
    Args:
        n_rows: Number of rows to generate
        
    Returns:
        pandas DataFrame with sample data
    """
    np.random.seed(42)
    
    # Date range for the past year
    dates = pd.date_range(end=pd.Timestamp.now(), periods=365)
    
    # Categories and regions
    categories = ["Category A", "Category B", "Category C", "Category D", "Category E"]
    regions = ["North", "South", "East", "West", "Central"]
    
    # Generate random data
    data = {
        "date": np.random.choice(dates, size=n_rows),
        "category": np.random.choice(categories, size=n_rows),
        "region": np.random.choice(regions, size=n_rows),
        "sales": np.random.randint(100, 10000, size=n_rows),
        "units": np.random.randint(1, 100, size=n_rows),
        "customer_id": np.random.randint(1000, 9999, size=n_rows)
    }
    
    df = pd.DataFrame(data)
    df["avg_price"] = (df["sales"] / df["units"]).round(2)
    
    return df

def get_time_series_chart(df, metric, aggregation="daily"):
    """
    Create a time series chart
    
    Args:
        df: DataFrame with time series data
        metric: Column to plot
        aggregation: Time aggregation (daily, weekly, monthly)
        
    Returns:
        Plotly figure
    """
    # Group by time period
    if aggregation == "daily":
        df_agg = df.groupby(df["date"].dt.date)[metric].mean().reset_index()
    elif aggregation == "weekly":
        df_agg = df.groupby(pd.Grouper(key="date", freq="W"))[metric].mean().reset_index()
    elif aggregation == "monthly":
        df_agg = df.groupby(pd.Grouper(key="date", freq="M"))[metric].mean().reset_index()
    else:  # quarterly
        df_agg = df.groupby(pd.Grouper(key="date", freq="Q"))[metric].mean().reset_index()
    
    # Create chart
    fig = px.line(
        df_agg, 
        x="date", 
        y=metric,
        markers=True,
        template="plotly_white"
    )
    
    fig.update_traces(
        line=dict(width=3, color="#1976D2"),
        marker=dict(size=8, color="#1976D2")
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="",
        yaxis_title=metric.replace("_", " ").title(),
        hovermode="x unified"
    )
    
    return fig

def get_correlation_chart(df, x_var, y_var, group_by=None):
    """
    Create a correlation scatter plot
    
    Args:
        df: DataFrame with data
        x_var: X-axis variable
        y_var: Y-axis variable
        group_by: Optional grouping variable
        
    Returns:
        Plotly figure and correlation coefficient
    """
    if group_by:
        fig = px.scatter(
            df,
            x=x_var,
            y=y_var,
            color=group_by,
            trendline="ols",
            template="plotly_white"
        )
    else:
        fig = px.scatter(
            df,
            x=x_var,
            y=y_var,
            trendline="ols",
            template="plotly_white"
        )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title=x_var.replace("_", " ").title(),
        yaxis_title=y_var.replace("_", " ").title()
    )
    
    # Calculate correlation
    corr = df[[x_var, y_var]].corr().iloc[0, 1]
    
    return fig, corr

def get_distribution_chart(df, var, bins=20, kde=True):
    """
    Create a distribution histogram
    
    Args:
        df: DataFrame with data
        var: Variable to plot
        bins: Number of bins
        kde: Whether to show density curve
        
    Returns:
        Plotly figure and summary statistics
    """
    fig = px.histogram(
        df,
        x=var,
        nbins=bins,
        marginal="box" if kde else None,
        template="plotly_white",
        color_discrete_sequence=["#1976D2"]
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title=var.replace("_", " ").title(),
        yaxis_title="Frequency"
    )
    
    # Calculate statistics
    stats = {
        "mean": df[var].mean(),
        "median": df[var].median(),
        "std": df[var].std(),
        "min": df[var].min(),
        "max": df[var].max(),
        "range": df[var].max() - df[var].min()
    }
    
    return fig, stats

def image_to_base64(img):
    """Convert PIL Image to base64 string for HTML embedding"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def create_gauge_chart(value, min_val=0, max_val=100, title="Metric"):
    """
    Create a gauge chart for displaying a single metric
    
    Args:
        value: Value to display
        min_val: Minimum value
        max_val: Maximum value
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": "#1976D2"},
            "steps": [
                {"range": [min_val, max_val * 0.3], "color": "#E3F2FD"},
                {"range": [max_val * 0.3, max_val * 0.7], "color": "#90CAF9"},
                {"range": [max_val * 0.7, max_val], "color": "#42A5F5"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    return fig

