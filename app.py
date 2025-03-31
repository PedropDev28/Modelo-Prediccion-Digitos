import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Data Science Dashboard",
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
    
    /* Metric styling */
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1976D2;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #616161;
        margin-top: 0.5rem;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #1976D2;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f1f3f4;
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
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .loading-animation {
        animation: pulse 1.5s infinite;
        background-color: #E3F2FD;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1976D2 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with improved styling
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/master/examples/data/logo.png", width=200)
    
    st.markdown("### Dashboard Settings")
    
    # Add a nice separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Date range selector with better UI
    st.markdown("#### Select Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
    with col2:
        end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))
    
    # Category filter with multi-select
    st.markdown("#### Filter Categories")
    categories = ["Category A", "Category B", "Category C", "Category D"]
    selected_categories = st.multiselect("Select Categories", categories, default=categories)
    
    # Value range slider
    st.markdown("#### Value Range")
    min_value, max_value = st.slider("Select Range", 0, 100, (25, 75))
    
    # Add a nice separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Apply button with enhanced styling
    st.markdown("""
    <div style="text-align: center; margin-top: 1rem;">
        <button style="
            background-color: #1976D2;
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 20px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        " onmouseover="this.style.backgroundColor='#1565C0'; this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.1)';"
        onmouseout="this.style.backgroundColor='#1976D2'; this.style.transform='translateY(0)'; this.style.boxShadow='none';">
            Apply Filters
        </button>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a nice separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # About section
    with st.expander("About", expanded=False):
        st.markdown("""
        This dashboard provides an interactive visualization of your data.
        
        **Features:**
        - Interactive charts and graphs
        - Data filtering and exploration
        - Downloadable reports
        - Real-time updates
        
        For more information, contact support@example.com
        """)

# Main content area
st.markdown("<h1 class='main-header'>Interactive Data Dashboard</h1>", unsafe_allow_html=True)

# Summary metrics in cards
st.markdown("<h2 class='sub-header'>Key Metrics</h2>", unsafe_allow_html=True)

# Create sample data for demonstration
np.random.seed(42)
metrics_data = {
    "Total Sales": f"${np.random.randint(100000, 999999):,}",
    "Customers": f"{np.random.randint(1000, 9999):,}",
    "Avg. Order Value": f"${np.random.randint(100, 999):,}",
    "Conversion Rate": f"{np.random.uniform(1, 5):.2f}%"
}

# Display metrics in a row of cards
cols = st.columns(4)
for i, (metric, value) in enumerate(metrics_data.items()):
    with cols[i]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{metric}</div>
        </div>
        """, unsafe_allow_html=True)

# Create tabs for different visualizations
st.markdown("<h2 class='sub-header'>Data Visualization</h2>", unsafe_allow_html=True)
tabs = st.tabs(["📈 Trends", "📊 Distribution", "🗺️ Geographic", "📋 Raw Data"])

# Sample data for charts
dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="M")
values = np.cumsum(np.random.randn(12)) + 20
df = pd.DataFrame({"Date": dates, "Value": values})

# Add some categorical data
categories = ["A", "B", "C", "D", "E"]
cat_values = np.random.randint(10, 100, size=len(categories))
cat_df = pd.DataFrame({"Category": categories, "Value": cat_values})

# Geographic data
states = ["California", "Texas", "Florida", "New York", "Pennsylvania"]
state_values = np.random.randint(100, 1000, size=len(states))
geo_df = pd.DataFrame({"State": states, "Value": state_values})

# Tab 1: Trends
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #1976D2;'>Monthly Trend Analysis</h3>", unsafe_allow_html=True)
    
    # Create a Plotly line chart
    fig = px.line(
        df, 
        x="Date", 
        y="Value",
        markers=True,
        line_shape="spline",
        template="plotly_white"
    )
    
    # Customize the chart
    fig.update_traces(
        line=dict(width=3, color="#1976D2"),
        marker=dict(size=8, color="#1976D2")
    )
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="",
        yaxis_title="",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add some context
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0;">Key Insights</h4>
            <ul>
                <li>Overall positive trend with 23% growth</li>
                <li>Seasonal peak in Q4</li>
                <li>Lowest performance in February</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Year-over-year comparison
        st.markdown("<h4>Year-over-Year Growth</h4>", unsafe_allow_html=True)
        yoy_metrics = {
            "Q1": 12.5,
            "Q2": 8.3,
            "Q3": 15.7,
            "Q4": 23.1
        }
        
        for quarter, growth in yoy_metrics.items():
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="width: 50px; font-weight: 500;">{quarter}</div>
                <div style="flex-grow: 1; margin: 0 10px;">
                    <div style="height: 8px; background-color: #e0e0e0; border-radius: 4px;">
                        <div style="height: 100%; width: {min(growth * 3, 100)}%; background-color: {'#4CAF50' if growth > 0 else '#F44336'}; border-radius: 4px;"></div>
                    </div>
                </div>
                <div style="width: 60px; text-align: right; color: {'#4CAF50' if growth > 0 else '#F44336'}; font-weight: 500;">
                    {'+' if growth > 0 else ''}{growth}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Distribution
with tabs[1]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #1976D2;'>Category Distribution</h3>", unsafe_allow_html=True)
    
    # Create a Plotly bar chart
    fig = px.bar(
        cat_df,
        x="Category",
        y="Value",
        color="Value",
        color_continuous_scale=px.colors.sequential.Blues,
        template="plotly_white"
    )
    
    # Customize the chart
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="",
        yaxis_title="",
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for percentage distribution
        labels = cat_df["Category"]
        values = cat_df["Value"]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker_colors=px.colors.sequential.Blues[::len(labels)]
        )])
        
        fig.update_layout(
            title="Percentage Distribution",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0;">Distribution Analysis</h4>
            <p>Category C shows the highest value, representing 32% of the total. Categories A and E are underperforming and may require additional attention.</p>
            <h5>Recommendations:</h5>
            <ul>
                <li>Increase marketing efforts for Categories A and E</li>
                <li>Analyze the success factors of Category C</li>
                <li>Consider reallocating resources based on performance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 3: Geographic
with tabs[2]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #1976D2;'>Geographic Distribution</h3>", unsafe_allow_html=True)
    
    # Create a US map
    fig = px.choropleth(
        geo_df,
        locations="State",
        locationmode="USA-states",
        color="Value",
        scope="usa",
        color_continuous_scale=px.colors.sequential.Blues,
        labels={"Value": "Sales"}
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        coloraxis_colorbar=dict(title="Sales")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add regional breakdown
    st.markdown("<h4>Regional Performance</h4>", unsafe_allow_html=True)
    
    # Create a horizontal bar chart for regions
    regions = ["West", "South", "Northeast", "Midwest"]
    region_values = np.random.randint(500, 2000, size=len(regions))
    region_df = pd.DataFrame({"Region": regions, "Value": region_values})
    
    fig = px.bar(
        region_df,
        y="Region",
        x="Value",
        orientation="h",
        color="Value",
        color_continuous_scale=px.colors.sequential.Blues,
        template="plotly_white"
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Sales",
        yaxis_title="",
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 4: Raw Data
with tabs[3]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #1976D2;'>Raw Data</h3>", unsafe_allow_html=True)
    
    # Create a larger sample dataset
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    categories = ["A", "B", "C", "D", "E"]
    regions = ["West", "South", "Northeast", "Midwest"]
    
    data = {
        "Date": np.random.choice(dates, size=100),
        "Category": np.random.choice(categories, size=100),
        "Region": np.random.choice(regions, size=100),
        "Sales": np.random.randint(100, 10000, size=100),
        "Units": np.random.randint(1, 100, size=100),
        "Customer ID": np.random.randint(1000, 9999, size=100)
    }
    
    raw_df = pd.DataFrame(data)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"]).dt.date
    raw_df["Avg Price"] = (raw_df["Sales"] / raw_df["Units"]).round(2)
    
    # Add search and filter options
    col1, col2 = st.columns(2)
    
    with col1:
        search_term = st.text_input("Search", placeholder="Enter search term...")
    
    with col2:
        filter_category = st.selectbox("Filter by Category", ["All"] + categories)
    
    # Apply filters
    filtered_df = raw_df.copy()
    
    if search_term:
        filtered_df = filtered_df[filtered_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)]
    
    if filter_category != "All":
        filtered_df = filtered_df[filtered_df["Category"] == filter_category]
    
    # Display the data with pagination
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download options
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="Download CSV",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name='data_export.csv',
            mime='text/csv',
        )
    
    with col2:
        st.download_button(
            label="Download Excel",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name='data_export.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Custom interactive component
st.markdown("<h2 class='sub-header'>Interactive Analysis</h2>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<h4>Configure Analysis</h4>", unsafe_allow_html=True)
    
    # Analysis options
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Time Series", "Correlation", "Distribution", "Forecast"]
    )
    
    # Parameters based on analysis type
    if analysis_type == "Time Series":
        metric = st.selectbox("Metric", ["Sales", "Units", "Avg Price"])
        aggregation = st.selectbox("Aggregation", ["Daily", "Weekly", "Monthly", "Quarterly"])
        include_trend = st.checkbox("Include Trend Line", value=True)
    
    elif analysis_type == "Correlation":
        x_var = st.selectbox("X-Axis", ["Sales", "Units", "Avg Price"])
        y_var = st.selectbox("Y-Axis", ["Units", "Sales", "Avg Price"])
        group_by = st.selectbox("Group By", ["None", "Category", "Region"])
    
    elif analysis_type == "Distribution":
        dist_var = st.selectbox("Variable", ["Sales", "Units", "Avg Price"])
        bins = st.slider("Number of Bins", 5, 50, 20)
        kde = st.checkbox("Show Density Curve", value=True)
    
    elif analysis_type == "Forecast":
        forecast_var = st.selectbox("Variable to Forecast", ["Sales", "Units", "Avg Price"])
        periods = st.slider("Forecast Periods", 7, 90, 30)
        confidence = st.slider("Confidence Interval", 70, 95, 80)
    
    # Run analysis button
    if st.button("Run Analysis", use_container_width=True):
        with st.spinner("Analyzing data..."):
            # Simulate processing time
            time.sleep(1.5)

with col2:
    # Display results based on selection
    if analysis_type == "Time Series" and 'metric' in locals():
        st.markdown(f"<h4>{metric} over Time ({aggregation})</h4>", unsafe_allow_html=True)
        
        # Generate sample time series data
        if aggregation == "Daily":
            dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")
        elif aggregation == "Weekly":
            dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="W")
        elif aggregation == "Monthly":
            dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="M")
        else:  # Quarterly
            dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="Q")
        
        values = np.cumsum(np.random.randn(len(dates))) + 20
        ts_df = pd.DataFrame({"Date": dates, "Value": values})
        
        # Create time series chart
        fig = px.line(
            ts_df, 
            x="Date", 
            y="Value",
            markers=True,
            template="plotly_white"
        )
        
        if include_trend:
            # Add trend line
            z = np.polyfit(range(len(ts_df)), ts_df["Value"], 1)
            p = np.poly1d(z)
            fig.add_scatter(
                x=ts_df["Date"],
                y=p(range(len(ts_df))),
                mode="lines",
                name="Trend",
                line=dict(color="red", dash="dash")
            )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="",
            yaxis_title=metric
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Correlation" and 'x_var' in locals() and 'y_var' in locals():
        st.markdown(f"<h4>Correlation: {x_var} vs {y_var}</h4>", unsafe_allow_html=True)
        
        # Generate sample correlation data
        n = 100
        x = np.random.normal(0, 1, n) * 1000 + 5000
        y = x * 0.5 + np.random.normal(0, 1, n) * 500 + 1000
        
        corr_df = pd.DataFrame({x_var: x, y_var: y})
        
        if group_by != "None":
            # Add grouping variable
            if group_by == "Category":
                groups = np.random.choice(["A", "B", "C", "D", "E"], size=n)
            else:  # Region
                groups = np.random.choice(["West", "South", "Northeast", "Midwest"], size=n)
            
            corr_df[group_by] = groups
            
            fig = px.scatter(
                corr_df,
                x=x_var,
                y=y_var,
                color=group_by,
                trendline="ols",
                template="plotly_white"
            )
        else:
            fig = px.scatter(
                corr_df,
                x=x_var,
                y=y_var,
                trendline="ols",
                template="plotly_white"
            )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation coefficient
        corr = np.corrcoef(x, y)[0, 1]
        st.markdown(f"""
        <div class="info-box">
            <p><strong>Correlation Coefficient:</strong> {corr:.2f}</p>
            <p><strong>Interpretation:</strong> {'Strong positive correlation' if corr > 0.7 else 'Moderate positive correlation' if corr > 0.3 else 'Weak correlation'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif analysis_type == "Distribution" and 'dist_var' in locals():
        st.markdown(f"<h4>Distribution of {dist_var}</h4>", unsafe_allow_html=True)
        
        # Generate sample distribution data
        if dist_var == "Sales":
            values = np.random.normal(5000, 1500, 1000)
        elif dist_var == "Units":
            values = np.random.poisson(30, 1000)
        else:  # Avg Price
            values = np.random.gamma(5, 30, 1000)
        
        dist_df = pd.DataFrame({dist_var: values})
        
        # Create histogram
        fig = px.histogram(
            dist_df,
            x=dist_var,
            nbins=bins,
            marginal="box" if kde else None,
            template="plotly_white",
            color_discrete_sequence=["#1976D2"]
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{np.mean(values):.2f}")
        col2.metric("Median", f"{np.median(values):.2f}")
        col3.metric("Std Dev", f"{np.std(values):.2f}")
        col4.metric("Range", f"{np.max(values) - np.min(values):.2f}")
    
    elif analysis_type == "Forecast" and 'forecast_var' in locals():
        st.markdown(f"<h4>Forecast: {forecast_var} (Next {periods} Days)</h4>", unsafe_allow_html=True)
        
        # Generate sample forecast data
        dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
        values = np.cumsum(np.random.randn(len(dates))) + 20
        
        # Create forecast dates
        forecast_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=periods, freq="D")
        
        # Create forecast values (simple trend continuation with noise)
        last_value = values[-1]
        trend = (values[-1] - values[-20]) / 20  # Calculate trend from last 20 days
        
        forecast_values = [last_value + trend * i + np.random.randn() * i * 0.2 for i in range(1, periods + 1)]
        
        # Create confidence intervals
        ci_factor = (100 - confidence) / 100
        lower_ci = [v - v * ci_factor * (i/periods) for i, v in enumerate(forecast_values)]
        upper_ci = [v + v * ci_factor * (i/periods) for i, v in enumerate(forecast_values)]
        
        # Combine historical and forecast data
        hist_df = pd.DataFrame({"Date": dates, "Value": values, "Type": "Historical"})
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Value": forecast_values,
            "Type": "Forecast",
            "Lower CI": lower_ci,
            "Upper CI": upper_ci
        })
        
        combined_df = pd.concat([hist_df, forecast_df])
        
        # Create forecast chart
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=hist_df["Date"],
            y=hist_df["Value"],
            mode="lines",
            name="Historical",
            line=dict(color="#1976D2")
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Value"],
            mode="lines",
            name="Forecast",
            line=dict(color="#FF9800", dash="dash")
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"].tolist() + forecast_df["Date"].tolist()[::-1],
            y=forecast_df["Upper CI"].tolist() + forecast_df["Lower CI"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(255, 152, 0, 0.2)",
            line=dict(color="rgba(255, 152, 0, 0)"),
            name=f"{confidence}% Confidence Interval"
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="",
            yaxis_title=forecast_var,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast metrics
        st.markdown(f"""
        <div class="info-box">
            <p><strong>Forecast Summary:</strong></p>
            <ul>
                <li>End of Period Forecast: {forecast_values[-1]:.2f}</li>
                <li>Average Forecast: {np.mean(forecast_values):.2f}</li>
                <li>Growth Rate: {((forecast_values[-1] / last_value) - 1) * 100:.2f}%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Add a footer
st.markdown("""
<div class="footer">
    <p>© 2023 Your Company Name. All rights reserved.</p>
    <p>Created with Streamlit • Last updated: December 2023</p>
</div>
""", unsafe_allow_html=True)

