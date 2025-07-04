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
            
            # Move Cluster Profiles Analysis to left column below Factor Analysis
            st.subheader("Cluster Profiles Analysis")
            
            # Create three columns for side-by-side cluster display
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
