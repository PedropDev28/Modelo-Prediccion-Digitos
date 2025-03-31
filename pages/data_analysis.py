import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_sample_data, get_time_series_chart, get_correlation_chart, get_distribution_chart

# Page configuration
st.set_page_config(
    page_title="Data Analysis",
    page_icon="📊",
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
st.markdown("<h1 class='main-header'>Advanced Data Analysis</h1>", unsafe_allow_html=True)

# Create sample data
if 'data' not in st.session_state:
    st.session_state.data = create_sample_data(1000)

df = st.session_state.data

# Sidebar with analysis options
with st.sidebar:
    st.markdown("### Analysis Options")
    
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Time Series", "Correlation", "Distribution", "Summary Statistics"]
    )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    if analysis_type == "Time Series":
        metric = st.selectbox(
            "Select Metric",
            ["sales", "units", "avg_price"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        aggregation = st.selectbox(
            "Time Aggregation",
            ["daily", "weekly", "monthly", "quarterly"],
            format_func=lambda x: x.title()
        )
        
        include_trend = st.checkbox("Include Trend Line", value=True)
        
    elif analysis_type == "Correlation":
        x_var = st.selectbox(
            "X-Axis Variable",
            ["sales", "units", "avg_price"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        y_var = st.selectbox(
            "Y-Axis Variable",
            ["units", "sales", "avg_price"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        group_by = st.selectbox(
            "Group By",
            ["None", "category", "region"],
            format_func=lambda x: x.title() if x != "None" else x
        )
        
    elif analysis_type == "Distribution":
        dist_var = st.selectbox(
            "Variable",
            ["sales", "units", "avg_price"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        bins = st.slider("Number of Bins", 5, 50, 20)
        kde = st.checkbox("Show Density Curve", value=True)
        
    elif analysis_type == "Summary Statistics":
        group_var = st.selectbox(
            "Group By",
            ["None", "category", "region"],
            format_func=lambda x: x.title() if x != "None" else x
        )
        
        metrics = st.multiselect(
            "Select Metrics",
            ["sales", "units", "avg_price"],
            default=["sales", "units"],
            format_func=lambda x: x.replace("_", " ").title()
        )

# Main content based on selected analysis
st.markdown("<div class='card'>", unsafe_allow_html=True)

if analysis_type == "Time Series":
    st.markdown(f"<h3 class='sub-header'>{metric.replace('_', ' ').title()} Over Time ({aggregation.title()})</h3>", unsafe_allow_html=True)
    
    # Create time series chart
    fig = get_time_series_chart(df, metric, aggregation)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate some basic stats for insights
        if aggregation == "daily":
            df_agg = df.groupby(df["date"].dt.date)[metric].mean().reset_index()
        elif aggregation == "weekly":
            df_agg = df.groupby(pd.Grouper(key="date", freq="W"))[metric].mean().reset_index()
        elif aggregation == "monthly":
            df_agg = df.groupby(pd.Grouper(key="date", freq="M"))[metric].mean().reset_index()
        else:  # quarterly
            df_agg = df.groupby(pd.Grouper(key="date", freq="Q"))[metric].mean().reset_index()
        
        recent_avg = df_agg[metric].iloc[-5:].mean()
        overall_avg = df_agg[metric].mean()
        percent_change = ((recent_avg / overall_avg) - 1) * 100
        
        st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0;">Key Insights</h4>
            <ul>
                <li>Recent trend shows {trend_direction} pattern</li>
                <li>Recent average is {percent_change:.1f}% {comparison} overall average</li>
                <li>Highest value observed on {max_date}</li>
                <li>Lowest value observed on {min_date}</li>
            </ul>
        </div>
        """.format(
            trend_direction="an increasing" if percent_change > 0 else "a decreasing",
            percent_change=abs(percent_change),
            comparison="above" if percent_change > 0 else "below",
            max_date=df_agg.loc[df_agg[metric].idxmax(), "date"].strftime("%B %d, %Y"),
            min_date=df_agg.loc[df_agg[metric].idxmin(), "date"].strftime("%B %d, %Y")
        ), unsafe_allow_html=True)
    
    with col2:
        # Period comparison
        st.markdown("<h4>Period Comparison</h4>", unsafe_allow_html=True)
        
        if aggregation == "monthly" or aggregation == "quarterly":
            # Compare last 4 periods
            last_periods = df_agg.tail(4)
            prev_periods = df_agg.iloc[-8:-4]
            
            for i, (curr, prev) in enumerate(zip(last_periods[metric], prev_periods[metric])):
                period_change = ((curr / prev) - 1) * 100
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="width: 80px; font-weight: 500;">{last_periods['date'].iloc[i].strftime('%b %Y')}</div>
                    <div style="flex-grow: 1; margin: 0 10px;">
                        <div style="height: 8px; background-color: #e0e0e0; border-radius: 4px;">
                            <div style="height: 100%; width: {min(abs(period_change) * 2, 100)}%; background-color: {'#4CAF50' if period_change > 0 else '#F44336'}; border-radius: 4px;"></div>
                        </div>
                    </div>
                    <div style="width: 60px; text-align: right; color: {'#4CAF50' if period_change > 0 else '#F44336'}; font-weight: 500;">
                        {'+' if period_change > 0 else ''}{period_change:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # For daily/weekly, show week-over-week or day-over-day
            periods = 7 if aggregation == "daily" else 4
            last_periods = df_agg.tail(periods)
            
            for i in range(len(last_periods)):
                if i == 0:
                    continue
                curr = last_periods[metric].iloc[i]
                prev = last_periods[metric].iloc[i-1]
                period_change = ((curr / prev) - 1) * 100
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="width: 80px; font-weight: 500;">{last_periods['date'].iloc[i].strftime('%b %d')}</div>
                    <div style="flex-grow: 1; margin: 0 10px;">
                        <div style="height: 8px; background-color: #e0e0e0; border-radius: 4px;">
                            <div style="height: 100%; width: {min(abs(period_change) * 2, 100)}%; background-color: {'#4CAF50' if period_change > 0 else '#F44336'}; border-radius: 4px;"></div>
                        </div>
                    </div>
                    <div style="width: 60px; text-align: right; color: {'#4CAF50' if period_change > 0 else '#F44336'}; font-weight: 500;">
                        {'+' if period_change > 0 else ''}{period_change:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

elif analysis_type == "Correlation":
    st.markdown(f"<h3 class='sub-header'>Correlation Analysis: {x_var.replace('_', ' ').title()} vs {y_var.replace('_', ' ').title()}</h3>", unsafe_allow_html=True)
    
    # Create correlation chart
    group_by_var = None if group_by == "None" else group_by
    fig, corr = get_correlation_chart(df, x_var, y_var, group_by_var)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add correlation interpretation
    st.markdown(f"""
    <div class="info-box">
        <h4 style="margin-top: 0;">Correlation Analysis</h4>
        <p><strong>Correlation Coefficient:</strong> {corr:.2f}</p>
        <p><strong>Interpretation:</strong> {
            'Strong positive correlation' if corr > 0.7 else
            'Moderate positive correlation' if corr > 0.3 else
            'Weak positive correlation' if corr > 0 else
            'Strong negative correlation' if corr < -0.7 else
            'Moderate negative correlation' if corr < -0.3 else
            'Weak negative correlation'
        }</p>
        <p><strong>Explanation:</strong> {
            'These variables show a strong direct relationship. As one increases, the other tends to increase significantly.' if corr > 0.7 else
            'These variables show a moderate direct relationship. As one increases, the other tends to increase somewhat.' if corr > 0.3 else
            'These variables show a weak direct relationship. There is a slight tendency for one to increase as the other increases.' if corr > 0 else
            'These variables show a strong inverse relationship. As one increases, the other tends to decrease significantly.' if corr < -0.7 else
            'These variables show a moderate inverse relationship. As one increases, the other tends to decrease somewhat.' if corr < -0.3 else
            'These variables show a weak inverse relationship. There is a slight tendency for one to decrease as the other increases.'
        }</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add scatter plots by category if grouping is used
    if group_by != "None":
        st.markdown(f"<h4>Breakdown by {group_by.title()}</h4>", unsafe_allow_html=True)
        
        # Calculate correlation by group
        group_corrs = df.groupby(group_by)[[x_var, y_var]].corr().iloc[::2, 1].reset_index()
        group_corrs.columns = [group_by, 'corr']
        
        # Create bar chart of correlations by group
        fig = px.bar(
            group_corrs,
            x=group_by,
            y='corr',
            color='corr',
            color_continuous_scale=px.colors.diverging.RdBu_r,
            template="plotly_white"
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Correlation Coefficient",
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Distribution":
    st.markdown(f"<h3 class='sub-header'>Distribution Analysis: {dist_var.replace('_', ' ').title()}</h3>", unsafe_allow_html=True)
    
    # Create distribution chart
    fig, stats = get_distribution_chart(df, dist_var, bins, kde)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{stats['mean']:.2f}")
    col2.metric("Median", f"{stats['median']:.2f}")
    col3.metric("Std Dev", f"{stats['std']:.2f}")
    col4.metric("Range", f"{stats['range']:.2f}")
    
    # Add distribution analysis
    st.markdown(f"""
    <div class="info-box">
        <h4 style="margin-top: 0;">Distribution Analysis</h4>
        <p><strong>Shape:</strong> {
            'Normally distributed' if abs((stats['mean'] - stats['median']) / stats['std']) < 0.2 else
            'Right-skewed (positive skew)' if stats['mean'] > stats['median'] else
            'Left-skewed (negative skew)'
        }</p>
        <p><strong>Spread:</strong> The data has a standard deviation of {stats['std']:.2f}, which is {
            'high' if stats['std'] / stats['mean'] > 0.5 else
            'moderate' if stats['std'] / stats['mean'] > 0.2 else
            'low'
        } relative to the mean.</p>
        <p><strong>Range:</strong> Values range from {stats['min']:.2f} to {stats['max']:.2f}, spanning a total of {stats['range']:.2f} units.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Distribution by category
    st.markdown("<h4>Distribution by Category</h4>", unsafe_allow_html=True)
    
    # Create box plots by category and region
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(
            df,
            x="category",
            y=dist_var,
            color="category",
            template="plotly_white",
            title="Distribution by Category"
        )
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            xaxis_title="",
            yaxis_title=dist_var.replace("_", " ").title()
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df,
            x="region",
            y=dist_var,
            color="region",
            template="plotly_white",
            title="Distribution by Region"
        )
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            xaxis_title="",
            yaxis_title=dist_var.replace("_", " ").title()
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Summary Statistics":
    st.markdown("<h3 class='sub-header'>Summary Statistics</h3>", unsafe_allow_html=True)
    
    if group_var == "None":
        # Overall summary statistics
        summary_df = df[metrics].describe().T.reset_index()
        summary_df.columns = ["Metric", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
        
        # Format the summary table
        for col in summary_df.columns[1:]:
            summary_df[col] = summary_df[col].round(2)
        
        st.dataframe(summary_df, use_container_width=True)
        
        # Create visualizations for each metric
        st.markdown("<h4>Metric Visualizations</h4>", unsafe_allow_html=True)
        
        for metric in metrics:
            st.markdown(f"<h5>{metric.replace('_', ' ').title()}</h5>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(
                    df,
                    x=metric,
                    nbins=20,
                    template="plotly_white"
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title=metric.replace("_", " ").title(),
                    yaxis_title="Frequency"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(
                    df,
                    y=metric,
                    template="plotly_white"
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=30, b=20),
                    yaxis_title=metric.replace("_", " ").title()
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        # Group by summary statistics
        st.markdown(f"<h4>Summary by {group_var.title()}</h4>", unsafe_allow_html=True)
        
        # Create summary table by group
        summary_list = []
        
        for group in df[group_var].unique():
            group_df = df[df[group_var] == group]
            
            for metric in metrics:
                summary = {
                    group_var: group,
                    "Metric": metric.replace("_", " ").title(),
                    "Count": len(group_df),
                    "Mean": group_df[metric].mean(),
                    "Median": group_df[metric].median(),
                    "Std": group_df[metric].std(),
                    "Min": group_df[metric].min(),
                    "Max": group_df[metric].max()
                }
                
                summary_list.append(summary)
        
        summary_df = pd.DataFrame(summary_list)
        
        # Format the summary table
        for col in ["Mean", "Median", "Std", "Min", "Max"]:
            summary_df[col] = summary_df[col].round(2)
        
        st.dataframe(summary_df, use_container_width=True)
        
        # Create visualizations for each metric
        for metric in metrics:
            st.markdown(f"<h5>{metric.replace('_', ' ').title()} by {group_var.title()}</h5>", unsafe_allow_html=True)
            
            # Bar chart by group
            fig = px.bar(
                df.groupby(group_var)[metric].mean().reset_index(),
                x=group_var,
                y=metric,
                color=group_var,
                template="plotly_white"
            )
            
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="",
                yaxis_title=metric.replace("_", " ").title(),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot by group
            fig = px.box(
                df,
                x=group_var,
                y=metric,
                color=group_var,
                template="plotly_white"
            )
            
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="",
                yaxis_title=metric.replace("_", " ").title(),
                showlegend=False
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

