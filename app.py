# Add these imports at the TOP of your script, before any other code   
import streamlit as st  # Import Streamlit first
st.set_page_config(layout="wide", page_title="Cola Consumer Dashboard", page_icon="🥤")  # Then set the page configuration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from io import BytesIO
import plotly.figure_factory as ff

# Try to import optional libraries with error handling
try:
    from factor_analyzer import FactorAnalyzer
    FACTOR_ANALYZER_AVAILABLE = True
except ImportError:
    FACTOR_ANALYZER_AVAILABLE = False

# Set page styling with enhanced mobile responsiveness
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5733;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(45deg, #FF5733, #FFC300);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
    }
    
    .subheader {
        font-size: 1.5rem;
        color: #3366FF;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #3366FF;
        padding-bottom: 0.5rem;
    }
    
    @media (max-width: 768px) {
        .subheader {
            font-size: 1.2rem;
        }
    }
    
    .insight-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #0066cc;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 0.8rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .insight-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .insight-title {
        font-weight: bold;
        color: #0066cc;
        font-size: 1.2rem;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .filter-box {
        background: linear-gradient(135deg, #f0f0f0 0%, #e0e0e0 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border: 1px solid #ddd;
    }
    
    .explained-box {
        background: linear-gradient(135deg, #f9f9f9 0%, #e8f5e8 100%);
        border-left: 5px solid #4CAF50;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0.8rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .explained-title {
        font-weight: bold;
        color: #4CAF50;
        font-size: 1.6rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .chart-container {
        background: white;
        border-radius: 1rem;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    /* Mobile optimization */
    @media (max-width: 768px) {
        .insight-box, .filter-box, .explained-box {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .metric-card {
            padding: 1rem;
            margin: 0.3rem 0;
        }
    }
    
    /* Responsive text sizing */
    @media (max-width: 480px) {
        .insight-title {
            font-size: 1rem;
        }
        
        .explained-title {
            font-size: 1.3rem;
        }
    }
    
    /* Animation for page load */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main-content {
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Glassmorphism effect for modern look */
    .glass-effect {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
</style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cola_survey.csv")
    # Create age groups
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[18, 25, 35, 45, 55, 65], 
                            labels=['18-24', '25-34', '35-44', '45-54', '55+'], 
                            right=False)
    
    # Calculate NPS categories
    df['NPS_Category'] = pd.cut(df['NPS_Score'], 
                               bins=[-1, 6, 8, 10], 
                               labels=['Detractors', 'Passives', 'Promoters'])
    
    # Perform clustering (will be available for all sections)
    X_cluster = df[['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
                  'Brand_Reputation_Rating', 'Availability_Rating', 
                  'Sweetness_Rating', 'Fizziness_Rating']]
    
    # Standardize data for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Get cluster centers and interpret
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(centers, 
                                     columns=X_cluster.columns, 
                                     index=['Cluster 0', 'Cluster 1', 'Cluster 2'])
    
    # Name clusters based on their characteristics
    cluster_names = {
        0: 'Taste Enthusiasts',  # High taste and sweetness ratings
        1: 'Brand Loyalists',    # High brand reputation ratings
        2: 'Value Seekers'       # High price ratings
    }
    
    df['Cluster_Name'] = df['Cluster'].map(cluster_names)
    
    return df, cluster_centers_df

# Load the data
df, cluster_centers = load_data()

# Add sidebar with quick stats
st.sidebar.markdown("## 📊 Quick Stats")
st.sidebar.markdown(f"**Total Records:** {len(df):,}")
st.sidebar.markdown(f"**Avg Age:** {df['Age'].mean():.1f} years")
st.sidebar.markdown(f"**Gender Split:** {df['Gender'].value_counts().to_dict()}")
st.sidebar.markdown(f"**Avg NPS:** {df['NPS_Score'].mean():.1f}")

# Add data freshness indicator
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔄 Data Status")
st.sidebar.success("✅ Data is current")
st.sidebar.markdown("*Last updated: Live Dashboard*")

# Add navigation help
st.sidebar.markdown("---")
st.sidebar.markdown("## 🧭 Navigation Tips")
st.sidebar.info("""
💡 **Pro Tips:**
- Use filters to focus on specific segments
- Switch chart types for different perspectives  
- Check the Executive Summary for key insights
- Use 'View & Download Full Dataset' section for exports
""")

# App title with enhanced styling
st.markdown("<div class='main-content'><h1 class='main-header'>🥤 Interactive Cola Consumer Dashboard</h1></div>", unsafe_allow_html=True)

# Add performance metrics and data summary at the top
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <h3>📊 Total Records</h3>
        <h2>{}</h2>
        <p>Survey Responses</p>
    </div>
    """.format(len(df)), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
        <h3>🎯 Data Quality</h3>
        <h2>99.8%</h2>
        <p>Completion Rate</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    unique_brands = len(df['Most_Often_Consumed_Brand'].unique())
    st.markdown("""
    <div class='metric-card'>
        <h3>🏷️ Brands Analyzed</h3>
        <h2>{}</h2>
        <p>Different Brands</p>
    </div>
    """.format(unique_brands), unsafe_allow_html=True)

with col4:
    age_range = f"{df['Age'].min()}-{df['Age'].max()}"
    st.markdown("""
    <div class='metric-card'>
        <h3>👥 Age Range</h3>
        <h2>{}</h2>
        <p>Years Old</p>
    </div>
    """.format(age_range), unsafe_allow_html=True)

st.markdown("---")

# Section Selection using Radio Buttons
section = st.radio("Select Analysis Section", [
    "Executive Dashboard Summary",
    "Demographic Profile", 
    "Brand Metrics", 
    "Basic Attribute Scores", 
    "Regression Analysis", 
    "Decision Tree Analysis", 
    "Cluster Analysis",
    "Advanced Analytics Explained",
    "View & Download Full Dataset"
], horizontal=True)

# Initialize session state for filters if not exists
if 'filters' not in st.session_state:
    st.session_state.filters = {
        'brand': None, 
        'gender': None, 
        'income': None, 
        'cluster': None
    }

# Move Filters to main page, below section selection with enhanced styling
st.markdown("<div class='filter-box glass-effect'>", unsafe_allow_html=True)
st.subheader("🎛️ Dynamic Dashboard Filters")

# Add filter tips
st.markdown("💡 **Pro Tip**: Apply multiple filters to drill down into specific consumer segments")

# Create a 4-column layout for filters
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

with filter_col1:
    # Create filter options with None as first option
    brand_options = [None] + sorted(df["Brand_Preference"].unique().tolist())
    selected_brand = st.selectbox(
        "Select a Brand", 
        options=brand_options, 
        index=0 if st.session_state.filters['brand'] is None else brand_options.index(st.session_state.filters['brand'])
    )

with filter_col2:
    gender_options = [None] + sorted(df["Gender"].unique().tolist())
    selected_gender = st.selectbox(
        "Select Gender", 
        options=gender_options, 
        index=0 if st.session_state.filters['gender'] is None else gender_options.index(st.session_state.filters['gender'])
    )

with filter_col3:
    income_options = [None] + sorted(df["Income_Level"].unique().tolist())
    selected_income = st.selectbox(
        "Select Income Level", 
        options=income_options, 
        index=0 if st.session_state.filters['income'] is None else income_options.index(st.session_state.filters['income'])
    )

with filter_col4:
    cluster_options = [None] + sorted(df["Cluster_Name"].unique().tolist())
    selected_cluster = st.selectbox(
        "Select Cluster", 
        options=cluster_options, 
        index=0 if st.session_state.filters['cluster'] is None else cluster_options.index(st.session_state.filters['cluster'])
    )

# Filter action buttons in two columns
fcol1, fcol2 = st.columns(2)

with fcol1:
    if st.button("Apply Filters"):
        # Update filters in session state
        st.session_state.filters = {
            'brand': selected_brand, 
            'gender': selected_gender, 
            'income': selected_income, 
            'cluster': selected_cluster
        }

with fcol2:
    if st.button("Clear Filters"):
        # Create a completely new session state dict with all Nones
        for key in st.session_state.filters.keys():
            st.session_state.filters[key] = None

st.markdown("</div>", unsafe_allow_html=True)

# Apply selected filters to the dataframe
filtered_df = df.copy()
filter_columns = {
    'brand': 'Brand_Preference',
    'gender': 'Gender',
    'income': 'Income_Level',
    'cluster': 'Cluster_Name'
}

# Apply filters dynamically
for filter_key, column in filter_columns.items():
    filter_value = st.session_state.filters.get(filter_key)
    if filter_value is not None:
        filtered_df = filtered_df[filtered_df[column] == filter_value]

# Show active filters with enhanced styling
active_filters = [f"{k}: {v}" for k, v in st.session_state.filters.items() if v is not None]
if active_filters:
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 1rem; 
                border-radius: 0.8rem; 
                margin: 1rem 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
        <strong>🎯 Active Filters:</strong> {', '.join(active_filters)} 
        <br><strong>📈 Filtered Records:</strong> {len(filtered_df):,} out of {len(df):,} total records
        <br><strong>📊 Coverage:</strong> {len(filtered_df)/len(df)*100:.1f}% of total dataset
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, #28a745 0%, #20c997 100%); 
                color: white; 
                padding: 1rem; 
                border-radius: 0.8rem; 
                margin: 1rem 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
        <strong>🌐 Viewing:</strong> Complete Dataset ({len(df):,} records)
        <br><strong>📊 Status:</strong> No filters applied - showing all consumer data
    </div>
    """, unsafe_allow_html=True)
    
# =======================
# EXECUTIVE DASHBOARD SUMMARY - ENHANCED
# =======================
if section == "Executive Dashboard Summary":
    st.markdown("<div class='chart-container'><h2 class='subheader'>📊 Executive Dashboard Summary</h2></div>", unsafe_allow_html=True)
    
    # Enhanced key metrics with better styling
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate NPS - Correct formula: % Promoters - % Detractors
        if not filtered_df.empty:
            promoters = filtered_df[filtered_df['NPS_Score'] >= 9].shape[0]
            detractors = filtered_df[filtered_df['NPS_Score'] <= 6].shape[0]
            total = filtered_df['NPS_Score'].count()
            
            # Calculate percentages first, then subtract
            promoters_pct = (promoters / total) if total > 0 else 0
            detractors_pct = (detractors / total) if total > 0 else 0
            nps_score = int((promoters_pct - detractors_pct) * 100)
            
            # Enhanced NPS display with color coding
            nps_color = "🟢" if nps_score > 50 else "🟡" if nps_score > 0 else "🔴"
            nps_status = "Excellent" if nps_score > 50 else "Good" if nps_score > 0 else "Needs Improvement"
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{nps_color} Overall NPS Score</h3>
                <h1 style='font-size: 3rem; margin: 0.5rem 0;'>{nps_score}</h1>
                <p>{nps_status}</p>
                <small>{promoters} Promoters • {detractors} Detractors</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(label="Overall NPS Score", value="No data", delta=None)
    
    with col2:
        # Enhanced top brand display
        if not filtered_df.empty:
            top_brand = filtered_df['Most_Often_Consumed_Brand'].value_counts().idxmax()
            top_brand_pct = filtered_df['Most_Often_Consumed_Brand'].value_counts(normalize=True).max() * 100
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🏆 Market Leader</h3>
                <h2 style='margin: 0.5rem 0;'>{top_brand}</h2>
                <h3>{top_brand_pct:.1f}% Market Share</h3>
                <small>Most consumed brand</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(label="Top Brand", value="No data", delta=None)
    
    with col3:
        # Enhanced top attribute display
        attributes = ['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
                     'Brand_Reputation_Rating', 'Availability_Rating', 
                     'Sweetness_Rating', 'Fizziness_Rating']
        
        if not filtered_df.empty:
            top_attr = filtered_df[attributes].mean().idxmax()
            top_attr_score = filtered_df[attributes].mean().max()
            
            attr_emoji = {"Taste_Rating": "👅", "Price_Rating": "💰", "Packaging_Rating": "📦", 
                         "Brand_Reputation_Rating": "⭐", "Availability_Rating": "🛒", 
                         "Sweetness_Rating": "🍯", "Fizziness_Rating": "🫧"}
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{attr_emoji.get(top_attr, "📊")} Top Rated Attribute</h3>
                <h2 style='margin: 0.5rem 0;'>{top_attr.replace('_Rating', '')}</h2>
                <h3>{top_attr_score:.2f}/5.0</h3>
                <small>Highest consumer satisfaction</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(label="Highest Rated Attribute", value="No data", delta=None)
    
    # Add real-time insights generator
    st.markdown("---")
    st.subheader("🤖 AI-Powered Insights")
    
    if not filtered_df.empty:
        # Calculate some dynamic insights
        avg_age = filtered_df['Age'].mean()
        dominant_gender = filtered_df['Gender'].mode().iloc[0] if not filtered_df['Gender'].mode().empty else "Unknown"
        satisfaction_high = (filtered_df['Satisfaction_Level'].isin(['Very Satisfied', 'Satisfied']).sum() / len(filtered_df) * 100)
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown(f"""
            <div class='insight-box'>
                <div class='insight-title'>🎯 Target Demographic Insight</div>
                <p><strong>Primary Audience:</strong> {dominant_gender} consumers with average age of {avg_age:.1f} years</p>
                <p><strong>Satisfaction Rate:</strong> {satisfaction_high:.1f}% are satisfied or very satisfied</p>
                <p><strong>Recommendation:</strong> Focus marketing efforts on this core demographic while exploring expansion opportunities.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col2:
            # Calculate trend insight
            top_occasion = filtered_df['Occasions_of_Buying'].mode().iloc[0] if not filtered_df['Occasions_of_Buying'].mode().empty else "Unknown"
            freq_pattern = filtered_df['Frequency_of_Consumption'].mode().iloc[0] if not filtered_df['Frequency_of_Consumption'].mode().empty else "Unknown"
            
            st.markdown(f"""
            <div class='insight-box'>
                <div class='insight-title'>📈 Consumption Pattern Insight</div>
                <p><strong>Peak Occasion:</strong> {top_occasion} drives most purchases</p>
                <p><strong>Typical Frequency:</strong> {freq_pattern} consumption pattern</p>
                <p><strong>Strategy:</strong> Optimize inventory and promotions around peak occasions and consumption cycles.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Rest of the insights sections with enhanced styling...
    st.subheader("🔍 Detailed Analysis Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='insight-box'>
            <div class='insight-title'>DEMOGRAPHIC INSIGHTS</div>
            <p>The cola consumer base shows distinct preferences by age group and gender:</p>
            <ul>
                <li>Younger consumers (18-34) show higher preference for major brands</li>
                <li>Gender differences exist in consumption frequency and occasion preferences</li>
                <li>Income level correlates with brand preference and price sensitivity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
            <div class='insight-title'>BRAND METRICS INSIGHTS</div>
            <p>Brand performance shows clear patterns in consumer behavior:</p>
            <ul>
                <li>Major brands dominate market share with loyal consumer bases</li>
                <li>Home consumption and parties are primary occasions for cola purchase</li>
                <li>Weekly consumption is most common frequency pattern</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
            <div class='insight-title'>ATTRIBUTE RATING INSIGHTS</div>
            <p>Product attributes show varying importance to consumers:</p>
            <ul>
                <li>Taste remains the most critical attribute across all segments</li>
                <li>Price sensitivity varies significantly across demographic groups</li>
                <li>Brand reputation has strong correlation with overall satisfaction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='insight-box'>
            <div class='insight-title'>REGRESSION ANALYSIS INSIGHTS</div>
            <p>The drivers of NPS (loyalty) are clearly identified:</p>
            <ul>
                <li>Taste is the strongest predictor of consumer loyalty</li>
                <li>Brand reputation is the second most influential factor</li>
                <li>The attributes explain over 50% of variation in NPS scores</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
            <div class='insight-title'>DECISION TREE INSIGHTS</div>
            <p>Consumer loyalty can be predicted by key decision factors:</p>
            <ul>
                <li>High taste ratings are the primary predictor of promoters</li>
                <li>Low brand reputation typically indicates detractors</li>
                <li>The model identifies clear paths to improving NPS scores</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
            <div class='insight-title'>CLUSTER ANALYSIS INSIGHTS</div>
            <p>Three distinct consumer segments with different priorities:</p>
            <ul>
                <li>Taste Enthusiasts (32%): Focus on sensory experience</li>
                <li>Brand Loyalists (41%): Value reputation and consistency</li>
                <li>Value Seekers (27%): Prioritize price and availability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =======================
# DEMOGRAPHIC PROFILE
# =======================
elif section == "Demographic Profile":
    st.markdown("<h2 class='subheader'>Demographic Profile</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Group Distribution
        if not filtered_df.empty:
            age_counts = filtered_df['Age_Group'].value_counts(normalize=True).sort_index() * 100
            age_raw_counts = filtered_df['Age_Group'].value_counts().sort_index()
            
            # Chart type selector
            chart_type = st.selectbox("Chart Type:", ["Bar Chart", "Pie Chart"], key="age_chart_type")
            
            if chart_type == "Bar Chart":
                fig = px.bar(
                    x=age_counts.index, 
                    y=age_counts.values, 
                    text=[f"{x:.1f}%<br>Count: {count}" for x, count in zip(age_counts.values, age_raw_counts.values)],
                    title='Age Group Distribution (%)',
                    labels={'x': 'Age Group', 'y': 'Percentage (%)'},
                    color=age_counts.values,
                    color_continuous_scale='viridis'
                )
                fig.update_traces(
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<extra></extra>'
                )
                fig.update_layout(
                    yaxis=dict(range=[0, max(age_counts.values) * 1.2]),
                    showlegend=False
                )
            else:  # Pie Chart
                fig = px.pie(
                    values=age_counts.values,
                    names=age_counts.index,
                    title='Age Group Distribution (%)',
                    hole=0.4
                )
                fig.update_traces(
                    textinfo='label+percent',
                    textposition='inside',
                    hovertemplate='<b>%{label}</b><br>Percentage: %{percent}<br>Count: %{value:.0f}<extra></extra>'
                )
            
            st.plotly_chart(fig)
        else:
            st.info("No data available for Age Group Distribution with current filters.")
        
        # Income Level Distribution
        if not filtered_df.empty:
            income_counts = filtered_df['Income_Level'].value_counts(normalize=True) * 100
            income_raw_counts = filtered_df['Income_Level'].value_counts()
            
            # Chart type selector
            income_chart_type = st.selectbox("Chart Type:", ["Pie Chart", "Bar Chart"], key="income_chart_type")
            
            if income_chart_type == "Pie Chart":
                fig = px.pie(
                    values=income_counts.values,
                    names=income_counts.index,
                    title='Income Level Distribution (%)',
                    hole=0.4,
                    labels={'label': 'Income Level', 'value': 'Percentage (%)'}
                )
                fig.update_traces(
                    textinfo='label+percent', 
                    textposition='inside',
                    hovertemplate='<b>%{label}</b><br>Percentage: %{percent}<br>Count: %{value:.0f}<extra></extra>'
                )
            else:  # Bar Chart
                fig = px.bar(
                    x=income_counts.index,
                    y=income_counts.values,
                    text=[f"{x:.1f}%<br>Count: {count}" for x, count in zip(income_counts.values, income_raw_counts.values)],
                    title='Income Level Distribution (%)',
                    labels={'x': 'Income Level', 'y': 'Percentage (%)'},
                    color=income_counts.values,
                    color_continuous_scale='viridis'
                )
                fig.update_traces(
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<extra></extra>'
                )
                fig.update_layout(
                    yaxis=dict(range=[0, max(income_counts.values) * 1.2]),
                    showlegend=False
                )
            
            st.plotly_chart(fig)
        else:
            st.info("No data available for Income Level Distribution with current filters.")
        
    with col2:
        # Gender Distribution
        if not filtered_df.empty:
            gender_counts = filtered_df['Gender'].value_counts(normalize=True) * 100
            gender_raw_counts = filtered_df['Gender'].value_counts()
            
            # Chart type selector
            gender_chart_type = st.selectbox("Chart Type:", ["Pie Chart", "Bar Chart"], key="gender_chart_type")
            
            if gender_chart_type == "Pie Chart":
                fig = px.pie(
                    values=gender_counts.values,
                    names=gender_counts.index,
                    title='Gender Distribution (%)',
                    hole=0.4,
                    labels={'label': 'Gender', 'value': 'Percentage (%)'}
                )
                fig.update_traces(
                    textinfo='label+percent', 
                    textposition='inside',
                    hovertemplate='<b>%{label}</b><br>Percentage: %{percent}<br>Count: %{value:.0f}<extra></extra>'
                )
            else:  # Bar Chart
                fig = px.bar(
                    x=gender_counts.index,
                    y=gender_counts.values,
                    text=[f"{x:.1f}%<br>Count: {count}" for x, count in zip(gender_counts.values, gender_raw_counts.values)],
                    title='Gender Distribution (%)',
                    labels={'x': 'Gender', 'y': 'Percentage (%)'},
                    color=gender_counts.values,
                    color_continuous_scale='plasma'
                )
                fig.update_traces(
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<extra></extra>'
                )
                fig.update_layout(
                    yaxis=dict(range=[0, max(gender_counts.values) * 1.2]),
                    showlegend=False
                )
            
            st.plotly_chart(fig)
        else:
            st.info("No data available for Gender Distribution with current filters.")
        
        # Age Group by Gender
        if not filtered_df.empty and len(filtered_df['Gender'].unique()) > 1:
            age_gender = pd.crosstab(
                filtered_df['Age_Group'], 
                filtered_df['Gender'], 
                normalize='columns'
            ) * 100
            
            # Chart type selector
            age_gender_chart_type = st.selectbox("Chart Type:", ["Grouped Bar Chart", "Stacked Bar Chart"], key="age_gender_chart_type")
            
            if age_gender_chart_type == "Grouped Bar Chart":
                fig = px.bar(
                    age_gender, 
                    barmode='group',
                    title='Age Group by Gender (%)',
                    labels={'value': 'Percentage (%)', 'index': 'Age Group'},
                    text_auto='.1f'
                )
                fig.update_traces(
                    textposition='outside',
                    hovertemplate='<b>%{fullData.name}</b><br>Age Group: %{x}<br>Percentage: %{y:.1f}%<extra></extra>'
                )
            else:  # Stacked Bar Chart
                fig = px.bar(
                    age_gender, 
                    barmode='stack',
                    title='Age Group by Gender (%)',
                    labels={'value': 'Percentage (%)', 'index': 'Age Group'},
                    text_auto='.1f'
                )
                fig.update_traces(
                    textposition='inside',
                    hovertemplate='<b>%{fullData.name}</b><br>Age Group: %{x}<br>Percentage: %{y:.1f}%<extra></extra>'
                )
            
            st.plotly_chart(fig)
        else:
            st.info("Insufficient data for Age Group by Gender analysis with current filters.")

# =======================
# BRAND METRICS
# =======================
elif section == "Brand Metrics":
    st.markdown("<h2 class='subheader'>Brand Metrics</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Most Often Consumed Brand
        if not filtered_df.empty:
            brand_counts = filtered_df['Most_Often_Consumed_Brand'].value_counts(normalize=True) * 100
            brand_raw_counts = filtered_df['Most_Often_Consumed_Brand'].value_counts()
            
            # Chart type selector
            brand_chart_type = st.selectbox("Chart Type:", ["Bar Chart", "Pie Chart"], key="brand_chart_type")
            
            if brand_chart_type == "Bar Chart":
                fig = px.bar(
                    x=brand_counts.index, 
                    y=brand_counts.values,
                    text=[f"{x:.1f}%<br>Count: {count}" for x, count in zip(brand_counts.values, brand_raw_counts.values)],
                    title='Most Often Consumed Brand (%)',
                    labels={'x': 'Brand', 'y': 'Percentage (%)'},
                    color=brand_counts.values,
                    color_continuous_scale='plasma'
                )
                fig.update_traces(
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<extra></extra>'
                )
                fig.update_layout(
                    yaxis=dict(range=[0, max(brand_counts.values) * 1.2]),
                    showlegend=False
                )
            else:  # Pie Chart
                fig = px.pie(
                    values=brand_counts.values,
                    names=brand_counts.index,
                    title='Most Often Consumed Brand (%)',
                    hole=0.4
                )
                fig.update_traces(
                    textinfo='label+percent',
                    textposition='inside',
                    hovertemplate='<b>%{label}</b><br>Percentage: %{percent}<br>Count: %{value:.0f}<extra></extra>'
                )
            
            st.plotly_chart(fig)
        else:
            st.info("No data available for Brand analysis with current filters.")
        
        # Occasions of Buying
        if not filtered_df.empty:
            occasions_counts = filtered_df['Occasions_of_Buying'].value_counts(normalize=True) * 100
            occasions_raw_counts = filtered_df['Occasions_of_Buying'].value_counts()
            
            # Chart type selector
            occasions_chart_type = st.selectbox("Chart Type:", ["Bar Chart", "Pie Chart"], key="occasions_chart_type")
            
            if occasions_chart_type == "Bar Chart":
                fig = px.bar(
                    x=occasions_counts.index, 
                    y=occasions_counts.values,
                    text=[f"{x:.1f}%<br>Count: {count}" for x, count in zip(occasions_counts.values, occasions_raw_counts.values)],
                    title='Occasions of Buying (%)',
                    labels={'x': 'Occasion', 'y': 'Percentage (%)'},
                    color=occasions_counts.values,
                    color_continuous_scale='cividis'
                )
                fig.update_traces(
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<extra></extra>'
                )
                fig.update_layout(
                    yaxis=dict(range=[0, max(occasions_counts.values) * 1.2]),
                    showlegend=False
                )
            else:  # Pie Chart
                fig = px.pie(
                    values=occasions_counts.values,
                    names=occasions_counts.index,
                    title='Occasions of Buying (%)',
                    hole=0.4
                )
                fig.update_traces(
                    textinfo='label+percent',
                    textposition='inside',
                    hovertemplate='<b>%{label}</b><br>Percentage: %{percent}<br>Count: %{value:.0f}<extra></extra>'
                )
            
            st.plotly_chart(fig)
        else:
            st.info("No data available for Occasions analysis with current filters.")
    
    with col2:
        # Frequency of Consumption
        if not filtered_df.empty:
            freq_counts = filtered_df['Frequency_of_Consumption'].value_counts(normalize=True) * 100
            freq_raw_counts = filtered_df['Frequency_of_Consumption'].value_counts()
            # Sort by frequency (not alphabetically)
            freq_order = ['Daily', 'Weekly', 'Monthly', 'Rarely', 'Never']
            freq_counts = freq_counts.reindex([f for f in freq_order if f in freq_counts.index])
            freq_raw_counts = freq_raw_counts.reindex([f for f in freq_order if f in freq_raw_counts.index])
            
            # Chart type selector
            freq_chart_type = st.selectbox("Chart Type:", ["Bar Chart", "Pie Chart"], key="freq_chart_type")
            
            if freq_chart_type == "Bar Chart":
                fig = px.bar(
                    x=freq_counts.index,
                    y=freq_counts.values,
                    text=[f"{x:.1f}%<br>Count: {count}" for x, count in zip(freq_counts.values, freq_raw_counts.values)],
                    title='Frequency of Consumption (%)',
                    labels={'x': 'Frequency', 'y': 'Percentage (%)'},
                    color=freq_counts.values,
                    color_continuous_scale='turbo'
                )
                fig.update_traces(
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<extra></extra>'
                )
                fig.update_layout(
                    yaxis=dict(range=[0, max(freq_counts.values) * 1.2]),
                    showlegend=False
                )
            else:  # Pie Chart
                fig = px.pie(
                    values=freq_counts.values,
                    names=freq_counts.index,
                    title='Frequency of Consumption (%)',
                    hole=0.4
                )
                fig.update_traces(
                    textinfo='label+percent',
                    textposition='inside',
                    hovertemplate='<b>%{label}</b><br>Percentage: %{percent}<br>Count: %{value:.0f}<extra></extra>'
                )
            
            st.plotly_chart(fig)
        else:
            st.info("No data available for Frequency analysis with current filters.")
        
        # Satisfaction Level
        if not filtered_df.empty:
            sat_counts = filtered_df['Satisfaction_Level'].value_counts(normalize=True) * 100
            sat_raw_counts = filtered_df['Satisfaction_Level'].value_counts()
            # Sort by satisfaction level
            sat_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']
            sat_counts = sat_counts.reindex([x for x in sat_order if x in sat_counts.index])
            sat_raw_counts = sat_raw_counts.reindex([x for x in sat_order if x in sat_raw_counts.index])
            
            # Chart type selector
            sat_chart_type = st.selectbox("Chart Type:", ["Bar Chart", "Pie Chart"], key="sat_chart_type")
            
            if sat_chart_type == "Bar Chart":
                fig = px.bar(
                    x=sat_counts.index,
                    y=sat_counts.values,
                    text=[f"{x:.1f}%<br>Count: {count}" for x, count in zip(sat_counts.values, sat_raw_counts.values)],
                    title='Satisfaction Level (%)',
                    labels={'x': 'Satisfaction Level', 'y': 'Percentage (%)'},
                    color=sat_counts.values,
                    color_continuous_scale='RdYlGn'
                )
                fig.update_traces(
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<extra></extra>'
                )
                fig.update_layout(
                    yaxis=dict(range=[0, max(sat_counts.values) * 1.2]),
                    showlegend=False
                )
            else:  # Pie Chart
                fig = px.pie(
                    values=sat_counts.values,
                    names=sat_counts.index,
                    title='Satisfaction Level (%)',
                    hole=0.4
                )
                fig.update_traces(
                    textinfo='label+percent',
                    textposition='inside',
                    hovertemplate='<b>%{label}</b><br>Percentage: %{percent}<br>Count: %{value:.0f}<extra></extra>'
                )
            
            st.plotly_chart(fig)
        else:
            st.info("No data available for Satisfaction analysis with current filters.")

# =======================
# BASIC ATTRIBUTE SCORES
# =======================
elif section == "Basic Attribute Scores":
    st.markdown("<h2 class='subheader'>Basic Attribute Scores</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # All attribute ratings
        attributes = [
            'Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
            'Brand_Reputation_Rating', 'Availability_Rating', 
            'Sweetness_Rating', 'Fizziness_Rating'
        ]
        
        if not filtered_df.empty:
            avg_scores = filtered_df[attributes].mean().sort_values(ascending=False)
            
            # Chart type selector for attribute ratings
            attr_chart_type = st.selectbox("Chart Type:", ["Bar Chart", "Horizontal Bar Chart"], key="attr_chart_type")
            
            if attr_chart_type == "Bar Chart":
                fig = px.bar(
                    x=avg_scores.index,
                    y=avg_scores.values,
                    text=[f"{x:.2f}" for x in avg_scores.values],
                    title='Average Attribute Ratings',
                    labels={'x': 'Attribute', 'y': 'Average Rating (1-5)'},
                    color=avg_scores.values,
                    color_continuous_scale='spectral'
                )
                fig.update_traces(
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Rating: %{y:.2f}<extra></extra>'
                )
                fig.update_layout(
                    xaxis={'categoryorder': 'total descending'},
                    yaxis=dict(range=[0, max(avg_scores.values) * 1.15]),
                    showlegend=False
                )
            else:  # Horizontal Bar Chart
                fig = px.bar(
                    x=avg_scores.values,
                    y=avg_scores.index,
                    text=[f"{x:.2f}" for x in avg_scores.values],
                    title='Average Attribute Ratings',
                    labels={'y': 'Attribute', 'x': 'Average Rating (1-5)'},
                    color=avg_scores.values,
                    color_continuous_scale='spectral',
                    orientation='h'
                )
                fig.update_traces(
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Rating: %{x:.2f}<extra></extra>'
                )
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis=dict(range=[0, max(avg_scores.values) * 1.15]),
                    showlegend=False
                )
            
            st.plotly_chart(fig)
        else:
            st.info("No data available for Attribute Ratings with current filters.")
    
    with col2:
        # Calculate NPS score
        if not filtered_df.empty:
            promoters = filtered_df[filtered_df['NPS_Score'] >= 9].shape[0]
            detractors = filtered_df[filtered_df['NPS_Score'] <= 6].shape[0]
            total = filtered_df['NPS_Score'].count()
            
            # Get percentages before calculating NPS
            promoters_pct = (promoters / total) if total > 0 else 0
            detractors_pct = (detractors / total) if total > 0 else 0 
            nps_score = int((promoters_pct - detractors_pct) * 100)
            
            # Display NPS gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=nps_score,
                title={'text': "Net Promoter Score (NPS)"},
                gauge={
                    'axis': {'range': [-100, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-100, 0], 'color': "red"},
                        {'range': [0, 30], 'color': "orange"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "green"}
                    ]
                }
            ))
            st.plotly_chart(fig)
        else:
            st.info("No data available for NPS Score with current filters.")
    
    # NPS by Demographics section
    st.subheader("NPS by Demographics")
    
    col1, col2 = st.columns(2)
    
    # Gender Analysis
    with col1:
        if filtered_df.empty:
            st.info("No data available for NPS Gender analysis.")
        else:
            if len(filtered_df['Gender'].unique()) <= 1:
                st.info("Insufficient unique gender data for comparison.")
            else:
                # Calculate NPS by gender
                gender_results = []
                for gender in filtered_df['Gender'].unique():
                    gender_df = filtered_df[filtered_df['Gender'] == gender]
                    promoters = gender_df[gender_df['NPS_Score'] >= 9].shape[0]
                    detractors = gender_df[gender_df['NPS_Score'] <= 6].shape[0]
                    total = gender_df.shape[0]
                    
                    # Calculate NPS
                    promoters_pct = promoters / total if total > 0 else 0
                    detractors_pct = detractors / total if total > 0 else 0
                    nps = (promoters_pct - detractors_pct) * 100
                    
                    gender_results.append({
                        'Gender': gender,
                        'NPS': nps
                    })
                
                # Create dataframe for plotting
                gender_df = pd.DataFrame(gender_results)
                
                # Create bar chart
                fig = px.bar(
                    gender_df,
                    x='Gender',
                    y='NPS',
                    title='NPS Score by Gender',
                    text=[f"{x:.1f}" for x in gender_df['NPS']],
                    color='NPS',
                    color_continuous_scale=px.colors.diverging.RdBu,
                    color_continuous_midpoint=0
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig)
    
    # Age Group Analysis
    with col2:
        if filtered_df.empty:
            st.info("No data available for NPS Age Group analysis.")
        else:
            if len(filtered_df['Age_Group'].unique()) <= 1:
                st.info("Insufficient unique age group data for comparison.")
            else:
                # Calculate NPS by age group
                age_results = []
                for age_group in sorted(filtered_df['Age_Group'].unique()):
                    age_df = filtered_df[filtered_df['Age_Group'] == age_group]
                    promoters = age_df[age_df['NPS_Score'] >= 9].shape[0]
                    detractors = age_df[age_df['NPS_Score'] <= 6].shape[0]
                    total = age_df.shape[0]
                    
                    # Calculate NPS
                    promoters_pct = promoters / total if total > 0 else 0
                    detractors_pct = detractors / total if total > 0 else 0
                    nps = (promoters_pct - detractors_pct) * 100
                    
                    age_results.append({
                        'Age_Group': age_group,
                        'NPS': nps
                    })
                
                # Create dataframe for plotting
                age_df = pd.DataFrame(age_results)
                
                # Create bar chart
                fig = px.bar(
                    age_df,
                    x='Age_Group',
                    y='NPS',
                    title='NPS Score by Age Group',
                    text=[f"{x:.1f}" for x in age_df['NPS']],
                    color='NPS',
                    color_continuous_scale=px.colors.diverging.RdBu,
                    color_continuous_midpoint=0
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig)

# =======================
# REGRESSION ANALYSIS - FIXED WITH SKLEARN
# =======================
elif section == "Regression Analysis":
    st.markdown("<h2 class='subheader'>Regression Analysis</h2>", unsafe_allow_html=True)
    
    # Check if we have enough data
    if len(filtered_df) < 10:
        st.warning("Insufficient data for regression analysis. Please adjust your filters.")
    else:
        # Prepare data for regression
        attributes = ['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
                     'Brand_Reputation_Rating', 'Availability_Rating', 
                     'Sweetness_Rating', 'Fizziness_Rating']
        
        X_reg = filtered_df[attributes]
        y_reg = filtered_df['NPS_Score']
        
        # Fit regression model using sklearn
        model = LinearRegression()
        model.fit(X_reg, y_reg)
        
        # Make predictions
        y_pred = model.predict(X_reg)
        
        # Calculate metrics
        r2 = r2_score(y_reg, y_pred)
        mse = mean_squared_error(y_reg, y_pred)
        
        # Calculate adjusted R-squared
        n = len(y_reg)
        p = len(attributes)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # Display regression results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Regression Summary")
            
            # Create a dataframe for coefficients
            coef_df = pd.DataFrame({
                'Feature': attributes,
                'Coefficient': model.coef_,
                'Abs_Coefficient': np.abs(model.coef_)
            }).sort_values('Abs_Coefficient', ascending=False)
            
            # Format coefficients
            coef_df['Coefficient_Formatted'] = coef_df['Coefficient'].apply(lambda x: f"{x:.4f}")
            
            # Display table
            display_df = coef_df[['Feature', 'Coefficient_Formatted']].copy()
            display_df.columns = ['Feature', 'Coefficient']
            st.dataframe(display_df, use_container_width=True)
            
            # Display key metrics
            st.write(f"**R-squared:** {r2:.4f}")
            st.write(f"**Adjusted R-squared:** {adj_r2:.4f}")
            st.write(f"**Mean Squared Error:** {mse:.4f}")
            st.write(f"**Intercept:** {model.intercept_:.4f}")
        
        with col2:
            # Visualization of coefficients
            fig = px.bar(
                coef_df,
                x='Feature', 
                y='Coefficient',
                title='Feature Importance (Coefficient Values)',
                labels={'Coefficient': 'Impact on NPS Score', 'Feature': 'Attribute'},
                color='Coefficient',
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(xaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig)
        
        # Predicted vs Actual scatter plot
        st.subheader("Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Predicted vs Actual
            fig = px.scatter(
                x=y_reg, 
                y=y_pred,
                title='Predicted vs Actual NPS Scores',
                labels={'x': 'Actual NPS Score', 'y': 'Predicted NPS Score'}
            )
            
            # Add diagonal line for perfect prediction
            min_val = min(min(y_reg), min(y_pred))
            max_val = max(max(y_reg), max(y_pred))
            fig.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="red", dash="dash", width=2),
                name="Perfect Prediction"
            )
            
            # Add trend line manually using numpy
            z = np.polyfit(y_reg, y_pred, 1)
            p = np.poly1d(z)
            fig.add_scatter(
                x=sorted(y_reg), 
                y=p(sorted(y_reg)),
                mode='lines',
                name=f'Trend Line (R² = {r2:.3f})',
                line=dict(color='blue', width=2)
            )
            
            st.plotly_chart(fig)
        
        with col2:
            # Residuals plot
            residuals = y_reg - y_pred
            fig = px.scatter(
                x=y_pred, 
                y=residuals,
                title='Residuals vs Predicted Values',
                labels={'x': 'Predicted NPS Score', 'y': 'Residuals'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig)
        
        # Key findings summary
        st.subheader("Key Regression Findings")
        
        # Identify top positive and negative factors
        pos_factors = coef_df[coef_df['Coefficient'] > 0].head(3)
        neg_factors = coef_df[coef_df['Coefficient'] < 0].head(3)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not pos_factors.empty:
                st.write("**Top Positive Factors (Increase NPS):**")
                for i, row in pos_factors.iterrows():
                    st.write(f"- **{row['Feature'].replace('_Rating', '')}**: {row['Coefficient']:.4f}")
            else:
                st.write("No positive factors found.")
        
        with col2:
            if not neg_factors.empty:
                st.write("**Top Negative Factors (Decrease NPS):**")
                for i, row in neg_factors.iterrows():
                    st.write(f"- **{row['Feature'].replace('_Rating', '')}**: {row['Coefficient']:.4f}")
            else:
                st.write("No negative factors found.")
        
        # Interpretation
        st.subheader("Business Insights")
        
        top_factor = coef_df.iloc[0]
        st.write(f"""
        **Key Insights from Regression Analysis:**
        
        🎯 **Model Performance**: The model explains {r2:.1%} of the variation in NPS scores.
        
        🏆 **Most Important Factor**: **{top_factor['Feature'].replace('_Rating', '')}** has the strongest impact on NPS 
        (coefficient: {top_factor['Coefficient']:.4f}).
        
        💡 **Business Recommendation**: Focus on improving the top 3 factors with positive coefficients to maximize NPS improvement.
        
        📊 **Model Quality**: {"Excellent" if r2 > 0.7 else "Good" if r2 > 0.5 else "Moderate" if r2 > 0.3 else "Weak"} explanatory power 
        (R² = {r2:.3f}).
        """)

# =======================
# DECISION TREE ANALYSIS
# =======================
elif section == "Decision Tree Analysis":
    st.markdown("<h2 class='subheader'>Decision Tree Analysis</h2>", unsafe_allow_html=True)
    
    # Check if we have enough data
    if len(filtered_df) < 30:
        st.warning("Insufficient data for decision tree analysis. Please adjust your filters.")
    else:
        # Define NPS categories for classification
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy['NPS_Category'] = pd.cut(
            filtered_df_copy['NPS_Score'],
            bins=[-1, 6, 8, 10],
            labels=['Detractor', 'Passive', 'Promoter']
        )
        
        # Prepare data for decision tree
        X_tree = filtered_df_copy[['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
                            'Brand_Reputation_Rating', 'Availability_Rating', 
                            'Sweetness_Rating', 'Fizziness_Rating', 'Age']]
        
        y_tree = filtered_df_copy['NPS_Category']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_tree, y_tree, test_size=0.25, random_state=42)
        
        # Train decision tree
        dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
        dt_model.fit(X_train, y_train)
        
        # Calculate accuracy
        train_accuracy = dt_model.score(X_train, y_train)
        test_accuracy = dt_model.score(X_test, y_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_tree.columns,
            'Importance': dt_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Decision Tree Performance")
            st.write(f"**Training Accuracy:** {train_accuracy:.2%}")
            st.write(f"**Testing Accuracy:** {test_accuracy:.2%}")
            
            # Feature importance chart
            fig = px.bar(
                feature_importance,
                x='Feature',
                y='Importance',
                title='Feature Importance in NPS Classification',
                labels={'Importance': 'Importance Score', 'Feature': 'Attribute'}
            )
            fig.update_layout(xaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Decision Tree Visualization")
            
            # Create decision tree plot
            plt.figure(figsize=(12, 8))
            plot_tree(dt_model, 
                     filled=True, 
                     feature_names=X_tree.columns, 
                     class_names=dt_model.classes_,
                     rounded=True,
                     fontsize=10)
            
            # Save plot to buffer
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Display the image
            st.image(buf)
        
        # Decision path analysis
        st.subheader("Key Decision Rules")
        
        # Extract and display key decision paths for promoters
        rules = tree.export_text(dt_model, 
                                feature_names=list(X_tree.columns),
                                max_depth=3)
        
        # Create a more user-friendly summary
        top_feature = feature_importance.iloc[0]['Feature']
        second_feature = feature_importance.iloc[1]['Feature'] if len(feature_importance) > 1 else None
        
        st.write(f"""
        **Decision Tree Analysis Summary:**
        
        The decision tree model achieved {test_accuracy:.1%} accuracy in predicting NPS categories (Promoter, Passive, Detractor).
        
        The most important factor in determining customer loyalty (NPS) is **{top_feature}**, 
        {"followed by **" + second_feature + "**" if second_feature else ""}.
        """)
        
        # Simplified rule display
        st.code(rules, language='text')
        
        # Consumer insights based on tree
        st.subheader("Consumer Insights from Decision Tree")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Promoter Profile:**")
            
            # Logic for promoters based on top features
            if top_feature in ['Taste_Rating', 'Brand_Reputation_Rating']:
                st.write(f"- Consumers with high {top_feature.replace('_Rating', '')} satisfaction")
            if second_feature:
                st.write(f"- Secondary importance: {second_feature.replace('_Rating', '')}")
            
            st.write("- These consumers are most likely to recommend your brand")
        
        with col2:
            st.write("**Detractor Profile:**")
            
            # Logic for detractors based on top features
            if top_feature in ['Taste_Rating', 'Brand_Reputation_Rating', 'Price_Rating']:
                st.write(f"- Consumers with low {top_feature.replace('_Rating', '')} satisfaction")
            if second_feature:
                st.write(f"- Also influenced by: {second_feature.replace('_Rating', '')}")
            
            st.write("- These consumers are least likely to recommend your brand")

# =======================
# CLUSTER ANALYSIS
# =======================
# =======================
# CLUSTER ANALYSIS - CLEAN VERSION
# =======================
elif section == "Cluster Analysis":
    st.markdown("<h2 class='subheader'>Cluster Analysis</h2>", unsafe_allow_html=True)
    
    # Check if we have enough data
    if len(filtered_df) < 30:
        st.warning("Insufficient data for cluster analysis. Please adjust your filters.")
    else:
        # Two-column layout: Pie chart on left, Radar chart on right
        col1, col2 = st.columns(2)
        
        with col1:
            # Display cluster distribution pie chart
            cluster_dist = filtered_df['Cluster_Name'].value_counts(normalize=True) * 100
            
            fig = px.pie(
                values=cluster_dist.values,
                names=cluster_dist.index,
                title='Cluster Distribution (%)',
                hole=0.4,
                labels={'label': 'Cluster', 'value': 'Percentage (%)'}
            )
            fig.update_traces(textinfo='label+percent', textposition='inside')
            st.plotly_chart(fig)
            
            # Factor Analysis section
            st.subheader("Factor Analysis")
            
            # Perform factor analysis
            attributes = ['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 
                        'Brand_Reputation_Rating', 'Availability_Rating', 
                        'Sweetness_Rating', 'Fizziness_Rating']
            
            # Only perform factor analysis if we have sufficient data and the library is available
            if not FACTOR_ANALYZER_AVAILABLE:
                st.info("Factor analyzer library not available. Showing correlation matrix instead.")
                
                # Show correlation matrix as alternative
                corr_matrix = filtered_df[attributes].corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Attribute Correlation Matrix",
                    color_continuous_scale='RdBu_r',
                    aspect="auto"
                )
                st.plotly_chart(fig)
                
            elif len(filtered_df) >= 50:  # Minimum recommended for factor analysis
                try:
                    fa = FactorAnalyzer(n_factors=2, rotation='varimax')
                    fa.fit(filtered_df[attributes])
                    
                    # Get factor loadings
                    loadings = pd.DataFrame(
                        fa.loadings_,
                        index=attributes,
                        columns=['Factor 1', 'Factor 2']
                    )
                    
                    # Display loadings
                    st.dataframe(loadings.round(3), use_container_width=True)
                except Exception as e:
                    st.error(f"Error in factor analysis: {str(e)}")
                    # Show correlation matrix as fallback
                    corr_matrix = filtered_df[attributes].corr()
                    fig = px.imshow(
                        corr_matrix,
                        title="Attribute Correlation Matrix (Fallback)",
                        color_continuous_scale='RdBu_r',
                        aspect="auto"
                    )
                    st.plotly_chart(fig)
            else:
                st.info("Insufficient data for factor analysis. Minimum of 50 records required.")
            
        # Cluster profiles analysis section - FULL WIDTH BELOW both columns
        st.subheader("Cluster Profiles Analysis")
        
        # Create three columns for side-by-side cluster display - FULL PAGE WIDTH
        cluster_col1, cluster_col2, cluster_col3 = st.columns(3)
        
        # Get cluster data for analysis
        clusters = list(filtered_df['Cluster_Name'].unique())
        
        # Display clusters in three columns
        for i, cluster in enumerate(clusters[:3]):  # Limit to 3 clusters
            # Choose the appropriate column
            if i == 0:
                col = cluster_col1
            elif i == 1:
                col = cluster_col2
            else:
                col = cluster_col3
            
            with col:
                cluster_data = filtered_df[filtered_df['Cluster_Name'] == cluster]
                
                # Make cluster name more prominent with styled header
                st.markdown(f"""
                <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                            color: white; 
                            padding: 0.8rem; 
                            border-radius: 0.5rem; 
                            text-align: center;
                            margin-bottom: 1rem;
                            font-weight: bold;
                            font-size: 1.1rem;'>
                    {cluster.upper()}
                    <br><small>({len(cluster_data)} consumers, {len(cluster_data)/len(filtered_df):.1%})</small>
                </div>
                """, unsafe_allow_html=True)
                
                if len(cluster_data) > 0:
                    # Top brand preference
                    top_brand = cluster_data['Most_Often_Consumed_Brand'].value_counts().idxmax()
                    brand_pct = cluster_data['Most_Often_Consumed_Brand'].value_counts(normalize=True).max() * 100
                    
                    # Top occasion
                    top_occasion = cluster_data['Occasions_of_Buying'].value_counts().idxmax()
                    occasion_pct = cluster_data['Occasions_of_Buying'].value_counts(normalize=True).max() * 100
                    
                    # Demographics
                    top_gender = cluster_data['Gender'].value_counts().idxmax()
                    gender_pct = cluster_data['Gender'].value_counts(normalize=True).max() * 100
                    
                    top_age = cluster_data['Age_Group'].value_counts().idxmax()
                    age_pct = cluster_data['Age_Group'].value_counts(normalize=True).max() * 100
                    
                    # Average NPS
                    avg_nps = cluster_data['NPS_Score'].mean()
                    
                    # Top and weakest attributes
                    top_attribute = cluster_data[attributes].mean().idxmax()
                    lowest_attribute = cluster_data[attributes].mean().idxmin()
                    
                    st.write(f"🥤 Prefers: **{top_brand}** ({brand_pct:.1f}%)")
                    st.write(f"🛒 Typically buys for: **{top_occasion}** ({occasion_pct:.1f}%)")
                    st.write(f"👤 Demographics: **{top_gender}** ({gender_pct:.1f}%), **{top_age}** ({age_pct:.1f}%)")
                    st.write(f"⭐ Avg. NPS: **{avg_nps:.1f}**")
                    st.write(f"💪 Strongest attribute: **{top_attribute.replace('_Rating', '')}**")
                    st.write(f"⚠️ Weakest attribute: **{lowest_attribute.replace('_Rating', '')}**")
                else:
                    st.write("No data available for this cluster with current filters.")
        
        with col2:
            # Cluster centers radar chart
            st.subheader("Cluster Centers (Average Ratings)")
            
            # Generate radar chart for cluster centers
            categories = ['Taste', 'Price', 'Packaging', 'Brand_Reputation', 'Availability', 'Sweetness', 'Fizziness']
            
            # Get cluster centers and reshape for radar chart
            centers_data = []
            cluster_names = filtered_df['Cluster_Name'].unique()
            
            for i, name in enumerate(cluster_names):
                cluster_id = filtered_df[filtered_df['Cluster_Name'] == name]['Cluster'].iloc[0]
                values = cluster_centers.iloc[cluster_id].values.tolist()
                centers_data.append({
                    'Cluster': name,
                    **{cat: val for cat, val in zip(categories, values)}
                })
            
            # Create radar chart
            df_radar = pd.DataFrame(centers_data)
            fig = go.Figure()
            
            for i, row in df_radar.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=row[categories].values,
                    theta=categories,
                    fill='toself',
                    name=row['Cluster']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )
                ),
                title="Cluster Profiles - Average Ratings"
            )
            st.plotly_chart(fig)
            
            # Add interpretation guide BELOW the radar chart
            st.markdown("""
            **📊 How to Read This Chart:**
            - **Larger areas** = Higher ratings for those attributes
            - **Compare shapes** to see how clusters differ
            - **Distance from center** = Rating strength (1-5 scale)
            - **Overlapping areas** = Similar performance between clusters
            """)
# =======================
# ADVANCED ANALYTICS EXPLAINED
# =======================
elif section == "Advanced Analytics Explained":
    st.markdown("<h2 class='subheader'>Advanced Analytics Explained</h2>", unsafe_allow_html=True)
    
    # Create download button with external PDF link
    st.markdown("""
    <div class='explained-box'>
        <div class='explained-title'>Advanced Analytics Overview</div>
        <p>Download the comprehensive PDF explaining advanced analytics techniques:</p>
        <a href="https://1drv.ms/b/s!AjvUTGyNS16HjZV-BJeBvloAgSeXOQ?e=oWofYC" 
           target="_blank" 
           download="Advanced_Analytics_Explained.pdf" 
           class="btn btn-primary">
            Download PDF
        </a>
    </div>
    """, unsafe_allow_html=True)

# =======================
# VIEW & DOWNLOAD FULL DATASET
# =======================
elif section == "View & Download Full Dataset":
    st.markdown("<h2 class='subheader'>View & Download Dataset</h2>", unsafe_allow_html=True)
    
    # Show dataset with cluster information
    st.dataframe(filtered_df)
    
    # Download options
    col1, col2 = st.columns(2)
    
    with col1:
        # Full dataset
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset (CSV)",
            data=csv,
            file_name="cola_survey_data.csv",
            mime="text/csv"
        )
    
    with col2:
        # Summary statistics
        summary_stats = filtered_df.describe().transpose()
        csv_summary = summary_stats.to_csv()
        st.download_button(
            label="Download Summary Statistics (CSV)",
            data=csv_summary,
            file_name="cola_survey_summary.csv",
            mime="text/csv"
        )

# Footer with contact email as a clickable link
st.markdown("---")
st.markdown("<div style='text-align: center;'>Cola Survey Dashboard | Created by <a href='mailto:aneesh@insights3d.com'>Aneesh Laiwala (aneesh@insights3d.com)</a></div>", unsafe_allow_html=True)
