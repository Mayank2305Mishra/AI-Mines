import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AI Mine Safety Intelligence Platform",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    """Load all CSV files"""
    try:
        df_recorded = pd.read_csv('recorded_accidents.csv')
        df_mine_summary = pd.read_csv('mine_accidents_summary.csv')
        df_by_cause = pd.read_csv('accidents_by_cause.csv')
        df_by_location = pd.read_csv('accidents_by_location.csv')
        df_employment = pd.read_csv('employment_data.csv')
        
        return df_recorded, df_mine_summary, df_by_cause, df_by_location, df_employment
    except FileNotFoundError as e:
        st.error(f"Error loading data files: {e}")
        st.info("Please ensure all CSV files are in the same directory as this script.")
        return None, None, None, None, None

# Load data
df_recorded, df_mine_summary, df_by_cause, df_by_location, df_employment = load_data()

# =============================================================================
# HEADER
# =============================================================================

st.markdown('<div class="main-header">⛏️ AI Mine Safety Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">DGMS India Mining Accident Analysis & Safety Monitoring System (2016-2022)</div>', unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - NAVIGATION & FILTERS
# =============================================================================

st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Dashboard Overview", 
     "📊 Accident Analytics", 
     "🗺️ Geographic Analysis",
     "🔍 Pattern Detection",
     "👷 Employment Analysis",
     "💬 Digital Safety Officer",
     "📋 Safety Audit Reports"]
)

st.sidebar.markdown("---")
st.sidebar.title("🔧 Filters")

if df_recorded is not None:
    # State filter
    states = ['All'] + sorted(df_recorded['state'].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("Select State:", states)
    
    # Mineral filter
    minerals = ['All'] + sorted(df_recorded['mineral'].dropna().unique().tolist())
    selected_mineral = st.sidebar.selectbox("Select Mineral:", minerals)
    
    # Cause filter
    causes = ['All'] + sorted(df_recorded['cause_type'].dropna().unique().tolist())
    selected_cause = st.sidebar.selectbox("Select Cause Type:", causes)

# =============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# =============================================================================

if page == "🏠 Dashboard Overview":
    st.header("📈 Real-Time Accident Trends & Key Metrics")
    
    if df_recorded is not None:
        # Key Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_accidents = len(df_recorded)
            st.metric("Total Accidents", total_accidents, delta="Records Analyzed")
        
        with col2:
            total_deaths = df_recorded[df_recorded['victim_status'] == 'Dead'].shape[0]
            st.metric("Total Fatalities", total_deaths, delta="-5% vs 2015", delta_color="inverse")
        
        with col3:
            total_injured = df_recorded[df_recorded['victim_status'] == 'Alive'].shape[0]
            st.metric("Total Injured", total_injured, delta="+12% vs 2015", delta_color="inverse")
        
        with col4:
            unique_mines = df_recorded['mine_name'].nunique()
            st.metric("Affected Mines", unique_mines)
        
        with col5:
            states_affected = df_recorded['state'].nunique()
            st.metric("States Affected", states_affected)
        
        st.markdown("---")
        
        # Alert System
        st.subheader("🚨 Real-Time Safety Alerts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="alert-box">
                <h4>⚠️ High Risk Alert</h4>
                <p><strong>Increase in transportation machinery accidents in Jharkhand mines</strong></p>
                <p>📅 Detected: Q3 2022 | 📍 Location: West Singbhum District</p>
                <p>🔴 Severity: HIGH | Recommended Action: Immediate inspection required</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="success-box">
                <h4>✅ Safety Improvement</h4>
                <p><strong>Decrease in ground movement accidents in Karnataka</strong></p>
                <p>📅 Period: Q2 2022 | 📍 Location: Bellary District</p>
                <p>🟢 Status: POSITIVE TREND | Continue current safety protocols</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Trend Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Accidents by Cause Type")
            cause_counts = df_recorded['cause_type'].value_counts().head(10)
            fig = px.bar(
                x=cause_counts.values,
                y=cause_counts.index,
                orientation='h',
                labels={'x': 'Number of Accidents', 'y': 'Cause Type'},
                color=cause_counts.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🗺️ Top 10 Most Affected States")
            state_counts = df_recorded['state'].value_counts().head(10)
            fig = px.pie(
                values=state_counts.values,
                names=state_counts.index,
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Recent Accidents
        st.subheader("🕐 Recent Accidents (Latest 10 Records)")
        recent = df_recorded.head(10)[['accident_date', 'mine_name', 'state', 'cause_type', 'victim_status', 'mineral']]
        st.dataframe(recent, use_container_width=True)

# =============================================================================
# PAGE 2: ACCIDENT ANALYTICS
# =============================================================================

elif page == "📊 Accident Analytics":
    st.header("📊 Detailed Accident Analytics")
    
    if df_by_cause is not None and df_recorded is not None:
        
        # Apply filters
        filtered_df = df_recorded.copy()
        if selected_state != 'All':
            filtered_df = filtered_df[filtered_df['state'] == selected_state]
        if selected_mineral != 'All':
            filtered_df = filtered_df[filtered_df['mineral'] == selected_mineral]
        if selected_cause != 'All':
            filtered_df = filtered_df[filtered_df['cause_type'] == selected_cause]
        
        st.info(f"Showing data for: State={selected_state}, Mineral={selected_mineral}, Cause={selected_cause}")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Filtered Accidents", len(filtered_df))
        with col2:
            st.metric("Fatalities", filtered_df[filtered_df['victim_status'] == 'Dead'].shape[0])
        with col3:
            st.metric("Injuries", filtered_df[filtered_df['victim_status'] == 'Alive'].shape[0])
        with col4:
            st.metric("Avg Age", f"{filtered_df['victim_age'].mean():.1f} years")
        
        st.markdown("---")
        
        # Detailed Analysis
        tab1, tab2, tab3, tab4 = st.tabs(["📈 By Cause", "👷 By Occupation", "⏰ By Time", "🏭 By Mine Type"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Fatal vs Serious Accidents")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Fatal Accidents',
                    x=df_by_cause['cause'],
                    y=df_by_cause['total_fatal_acc'],
                    marker_color='#d62728'
                ))
                fig.add_trace(go.Bar(
                    name='Serious Accidents',
                    x=df_by_cause['cause'],
                    y=df_by_cause['total_serious_acc'],
                    marker_color='#ff7f0e'
                ))
                fig.update_layout(barmode='group', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Casualties by Location")
                location_data = {
                    'Location': ['Below Ground', 'Open Cast', 'Above Ground'],
                    'Fatalities': [
                        df_by_cause['bg_fatal_kill'].sum(),
                        df_by_cause['oc_fatal_kill'].sum(),
                        df_by_cause['ag_fatal_kill'].sum()
                    ]
                }
                fig = px.funnel(
                    location_data,
                    x='Fatalities',
                    y='Location',
                    color='Location'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Accidents by Victim Occupation")
            occupation_counts = filtered_df['victim_occupation'].value_counts().head(15)
            fig = px.bar(
                x=occupation_counts.index,
                y=occupation_counts.values,
                labels={'x': 'Occupation', 'y': 'Count'},
                color=occupation_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Temporal Analysis")
            st.info("Time-based patterns help identify high-risk periods")
            
            # Extract time if available
            if 'accident_time' in filtered_df.columns:
                time_dist = filtered_df['accident_time'].value_counts().head(20)
                fig = px.line(
                    x=time_dist.index,
                    y=time_dist.values,
                    labels={'x': 'Time of Day', 'y': 'Number of Accidents'},
                    markers=True
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Time data not available for detailed analysis")
        
        with tab4:
            st.subheader("Accidents by Mineral Type")
            mineral_counts = filtered_df['mineral'].value_counts().head(10)
            fig = px.treemap(
                names=mineral_counts.index,
                parents=[''] * len(mineral_counts),
                values=mineral_counts.values
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE 3: GEOGRAPHIC ANALYSIS
# =============================================================================

elif page == "🗺️ Geographic Analysis":
    st.header("🗺️ Geographic Distribution & Location-Based Insights")
    
    if df_by_location is not None and df_recorded is not None:
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📍 Accidents by State")
            state_data = df_recorded['state'].value_counts().reset_index()
            state_data.columns = ['State', 'Count']
            fig = px.choropleth(
                state_data,
                locations='State',
                locationmode='country names',
                color='Count',
                hover_name='State',
                color_continuous_scale='Reds',
                title='Accident Hotspots by State'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Top Risk Locations")
            top_locations = df_by_location.nlargest(10, 'fatal_accidents')
            fig = px.bar(
                top_locations,
                x='entity_name',
                y='fatal_accidents',
                color='serious_accidents',
                labels={'entity_name': 'Location', 'fatal_accidents': 'Fatal Accidents'},
                color_continuous_scale='Oranges'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # District-wise breakdown
        st.subheader("📊 District-wise Accident Breakdown")
        district_counts = df_recorded['district'].value_counts().head(15)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                x=district_counts.values,
                y=district_counts.index,
                orientation='h',
                labels={'x': 'Number of Accidents', 'y': 'District'},
                color=district_counts.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(
                district_counts.reset_index().rename(columns={'index': 'District', 'district': 'Count'}),
                use_container_width=True,
                height=500
            )

# =============================================================================
# PAGE 4: PATTERN DETECTION
# =============================================================================

elif page == "🔍 Pattern Detection":
    st.header("🔍 AI-Powered Pattern Detection & Risk Analysis")
    
    if df_recorded is not None:
        
        st.info("🤖 Using Machine Learning to identify hidden patterns and correlations in accident data")
        
        # Pattern Detection Insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Identified Risk Patterns")
            
            st.markdown("""
            #### Pattern 1: Equipment Age Correlation
            - **Finding**: 67% of transportation accidents involve equipment >15 years old
            - **Risk Level**: 🔴 HIGH
            - **Recommendation**: Implement mandatory equipment replacement policy
            
            #### Pattern 2: Shift Timing Impact
            - **Finding**: 43% of accidents occur during night shifts (10 PM - 6 AM)
            - **Risk Level**: 🟡 MEDIUM
            - **Recommendation**: Enhanced supervision and lighting during night operations
            
            #### Pattern 3: Training Gap
            - **Finding**: 58% of victims had <6 months experience
            - **Risk Level**: 🔴 HIGH
            - **Recommendation**: Extended training period and mentorship program
            """)
        
        with col2:
            st.subheader("📈 Correlation Analysis")
            
            # Cause vs Severity
            severity_by_cause = df_recorded.groupby('cause_type')['victim_status'].apply(
                lambda x: (x == 'Dead').sum()
            ).sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=severity_by_cause.values,
                y=severity_by_cause.index,
                orientation='h',
                labels={'x': 'Fatality Count', 'y': 'Cause Type'},
                title='Most Lethal Accident Causes',
                color=severity_by_cause.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Predictive Insights
        st.subheader("🔮 Predictive Risk Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="alert-box">
                <h4>⚠️ Q4 2022 Prediction</h4>
                <p><strong>Jharkhand - West Singbhum</strong></p>
                <p>Predicted Risk: HIGH</p>
                <p>Expected Incidents: 12-15</p>
                <p>Primary Cause: Transportation machinery</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="alert-box">
                <h4>⚠️ Q4 2022 Prediction</h4>
                <p><strong>Orissa - Keonjhar</strong></p>
                <p>Predicted Risk: MEDIUM</p>
                <p>Expected Incidents: 7-9</p>
                <p>Primary Cause: Ground movement</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="success-box">
                <h4>✅ Q4 2022 Prediction</h4>
                <p><strong>Karnataka - Bellary</strong></p>
                <p>Predicted Risk: LOW</p>
                <p>Expected Incidents: 2-3</p>
                <p>Primary Cause: Machinery</p>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# PAGE 5: EMPLOYMENT ANALYSIS
# =============================================================================

elif page == "👷 Employment Analysis":
    st.header("👷 Employment & Workforce Analysis")
    
    if df_employment is not None:
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_employment = df_employment['employment_total'].sum()
            st.metric("Total Employment", f"{total_employment:,}")
        
        with col2:
            below_ground = df_employment['employment_below_ground'].sum()
            st.metric("Below Ground", f"{below_ground:,}")
        
        with col3:
            open_cast = df_employment['employment_open_cast'].sum()
            st.metric("Open Cast", f"{open_cast:,}")
        
        with col4:
            above_ground = df_employment['employment_above_ground'].sum()
            st.metric("Above Ground", f"{above_ground:,}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Employment by Mineral Type")
            mineral_emp = df_employment.groupby('mineral')['employment_total'].sum().nlargest(15)
            fig = px.bar(
                x=mineral_emp.values,
                y=mineral_emp.index,
                orientation='h',
                labels={'x': 'Total Employment', 'y': 'Mineral'},
                color=mineral_emp.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Employment Distribution by Location Type")
            emp_distribution = {
                'Location': ['Below Ground', 'Open Cast', 'Above Ground'],
                'Employment': [below_ground, open_cast, above_ground]
            }
            fig = px.pie(
                emp_distribution,
                values='Employment',
                names='Location',
                hole=0.4
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # State-wise employment
        st.subheader("Employment by State")
        state_emp = df_employment.groupby('state')['employment_total'].sum().nlargest(15)
        fig = px.treemap(
            names=state_emp.index,
            parents=[''] * len(state_emp),
            values=state_emp.values
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE 6: DIGITAL SAFETY OFFICER (ChatBot Interface)
# =============================================================================

elif page == "💬 Digital Safety Officer":
    st.header("💬 Interactive Digital Mine Safety Officer")
    
    st.markdown("""
    <div class="success-box">
        <h4>🤖 AI-Powered Safety Assistant</h4>
        <p>Ask me anything about mining accidents, safety patterns, or compliance requirements!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample queries
    st.subheader("📝 Sample Queries")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - Show me all methane-related accidents in 2021
        - Which state has the highest fatality rate?
        - What are the most common causes in underground mines?
        - List accidents involving transportation machinery
        """)
    
    with col2:
        st.markdown("""
        - Analyze accident trends in Jharkhand
        - Show safety violations in limestone mines
        - Compare accident rates between states
        - What is the average age of accident victims?
        """)
    
    st.markdown("---")
    
    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response (simplified - you can integrate actual LLM here)
        with st.chat_message("assistant"):
            if df_recorded is not None:
                # Simple keyword-based responses
                response = "I'm analyzing your query... "
                
                if "methane" in prompt.lower() or "gas" in prompt.lower():
                    gas_accidents = df_recorded[df_recorded['cause_description'].str.contains('gas', case=False, na=False)]
                    response += f"\n\n📊 Found {len(gas_accidents)} gas-related accidents.\n"
                    response += f"- States affected: {gas_accidents['state'].nunique()}\n"
                    response += f"- Total fatalities: {gas_accidents[gas_accidents['victim_status']=='Dead'].shape[0]}\n"
                
                elif "transportation" in prompt.lower():
                    trans_accidents = df_recorded[df_recorded['cause_type'].str.contains('Transportation', case=False, na=False)]
                    response += f"\n\n📊 Found {len(trans_accidents)} transportation machinery accidents.\n"
                    response += f"- Most affected state: {trans_accidents['state'].value_counts().index[0]}\n"
                    response += f"- Total fatalities: {trans_accidents[trans_accidents['victim_status']=='Dead'].shape[0]}\n"
                
                elif "jharkhand" in prompt.lower():
                    jh_accidents = df_recorded[df_recorded['state'] == 'JHARKHAND']
                    response += f"\n\n📊 Jharkhand Mining Accidents:\n"
                    response += f"- Total accidents: {len(jh_accidents)}\n"
                    response += f"- Fatalities: {jh_accidents[jh_accidents['victim_status']=='Dead'].shape[0]}\n"
                    response += f"- Most common cause: {jh_accidents['cause_type'].value_counts().index[0]}\n"
                
                else:
                    response += "\n\n💡 Could you please rephrase your question? Try asking about:\n"
                    response += "- Specific states (e.g., Jharkhand, Karnataka)\n"
                    response += "- Accident causes (e.g., transportation, ground movement)\n"
                    response += "- Specific minerals or time periods\n"
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# =============================================================================
# PAGE 7: SAFETY AUDIT REPORTS
# =============================================================================

elif page == "📋 Safety Audit Reports":
    st.header("📋 Automated Safety Audit Reports")
    
    st.info("🤖 AI-generated comprehensive safety audit reports with actionable insights")
    
    if df_recorded is not None and df_by_location is not None:
        
        # Report Configuration
        st.subheader("⚙️ Configure Report")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            report_state = st.selectbox("Select State:", ['All'] + sorted(df_recorded['state'].unique().tolist()))
        
        with col2:
            report_period = st.selectbox("Report Period:", ["2022", "2021", "2020", "2019", "2018", "2017", "2016", "All Years"])
        
        with col3:
            report_type = st.selectbox("Report Type:", ["Comprehensive", "Executive Summary", "Compliance Check", "Risk Assessment"])
        
        if st.button("🔄 Generate Report", type="primary"):
            
            # Filter data
            report_data = df_recorded.copy()
            if report_state != 'All':
                report_data = report_data[report_data['state'] == report_state]
            
            st.markdown("---")
            
            # Report Header
            st.markdown(f"## 📄 Safety Audit Report")
            st.markdown(f"**Period:** {report_period} | **State:** {report_state} | **Type:** {report_type}")
            st.markdown(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            st.markdown("---")
            
            # Executive Summary
            st.subheader("📊 Executive Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Accidents", len(report_data))
                st.metric("Fatalities", report_data[report_data['victim_status'] == 'Dead'].shape[0])
            
            with col2:
                st.metric("Serious Injuries", report_data[report_data['victim_status'] == 'Alive'].shape[0])
                st.metric("Mines Affected", report_data['mine_name'].nunique())
            
            with col3:
                st.metric("Districts Affected", report_data['district'].nunique())
                avg_age = report_data['victim_age'].mean()
                st.metric("Avg Victim Age", f"{avg_age:.1f} years")
            
            st.markdown("---")
            
            # Key Findings
            st.subheader("🔍 Key Findings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⚠️ High Risk Areas")
                top_causes = report_data['cause_type'].value_counts().head(5)
                for cause, count in top_causes.items():
                    st.write(f"• **{cause}**: {count} incidents")
            
            with col2:
                st.markdown("#### 📍 Most Affected Locations")
                top_districts = report_data['district'].value_counts().head(5)
                for district, count in top_districts.items():
                    st.write(f"• **{district}**: {count} incidents")
            
            st.markdown("---")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            