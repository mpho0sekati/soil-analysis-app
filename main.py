import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json
import os

class SoilAnalyzer:
    def __init__(self):
        self.load_crop_requirements()
        self.initialize_models()
        
    def load_crop_requirements(self):
        # Optimal ranges for different crops
        self.crop_requirements = {
            "Wheat": {"pH": (6.0, 7.0), "N": (20, 40), "P": (10, 20), "K": (150, 250)},
            "Corn": {"pH": (5.8, 7.0), "N": (25, 45), "P": (12, 25), "K": (170, 270)},
            "Soybeans": {"pH": (6.0, 7.5), "N": (15, 30), "P": (15, 30), "K": (130, 230)},
            "Rice": {"pH": (5.5, 6.5), "N": (30, 50), "P": (8, 16), "K": (140, 240)},
        }

    def initialize_models(self):
        # Initialize machine learning models for different predictions
        self.yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.train_models()

    def train_models(self):
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        X = pd.DataFrame({
            'pH': np.random.uniform(5.0, 8.0, n_samples),
            'N': np.random.uniform(10, 60, n_samples),
            'P': np.random.uniform(5, 35, n_samples),
            'K': np.random.uniform(100, 350, n_samples),
            'Rainfall': np.random.uniform(500, 1500, n_samples),
            'Temperature': np.random.uniform(15, 30, n_samples)
        })
        
        # Synthetic yield calculation with some noise
        y = (
            100 * (-(X['pH'] - 6.5)**2) +
            2 * X['N'] +
            3 * X['P'] +
            0.5 * X['K'] +
            0.1 * X['Rainfall'] +
            -10 * ((X['Temperature'] - 22)**2) +
            np.random.normal(0, 100, n_samples)
        )
        
        # Scale features and train model
        X_scaled = self.scaler.fit_transform(X)
        self.yield_model.fit(X_scaled, y)

    def predict_yield(self, soil_data):
        scaled_data = self.scaler.transform(soil_data)
        return self.yield_model.predict(scaled_data)[0]

    def get_recommendations(self, soil_data, crop):
        recommendations = []
        crop_reqs = self.crop_requirements[crop]
        
        # pH recommendations
        if soil_data['pH'].iloc[0] < crop_reqs['pH'][0]:
            recommendations.append({
                'type': 'warning',
                'message': f'pH is too low for {crop}. Consider adding lime to increase pH.',
                'action': 'Add agricultural lime at 2-3 tons per hectare'
            })
        elif soil_data['pH'].iloc[0] > crop_reqs['pH'][1]:
            recommendations.append({
                'type': 'warning',
                'message': f'pH is too high for {crop}. Consider adding sulfur to decrease pH.',
                'action': 'Add elemental sulfur at 500-1000 kg per hectare'
            })

        # Nutrient recommendations
        nutrients = {'N': 'Nitrogen', 'P': 'Phosphorus', 'K': 'Potassium'}
        for nut, name in nutrients.items():
            if soil_data[nut].iloc[0] < crop_reqs[nut][0]:
                recommendations.append({
                    'type': 'warning',
                    'message': f'Low {name} levels for {crop}.',
                    'action': f'Apply {name}-rich fertilizer at recommended rates'
                })

        return recommendations

class SoilAnalysisApp:
    def __init__(self):
        self.analyzer = SoilAnalyzer()
        self.setup_streamlit()

    def setup_streamlit(self):
        st.set_page_config(
            page_title="SoilSense Pro",
            page_icon="🌱",
            layout="wide"
        )
        
        # Initialize session state
        if 'soil_history' not in st.session_state:
            st.session_state.soil_history = pd.DataFrame()

    def run(self):
        st.title("🌱 SoilSense Pro")
        st.subheader("Advanced Soil Analysis Platform")

        # Create tabs for different sections
        tabs = st.tabs(["Soil Analysis", "Historical Data", "Reports", "Settings"])

        with tabs[0]:
            self.soil_analysis_tab()

        with tabs[1]:
            self.historical_data_tab()

        with tabs[2]:
            self.reports_tab()

        with tabs[3]:
            self.settings_tab()

    def soil_analysis_tab(self):
        col1, col2 = st.columns([2, 1])

        with col2:
            st.subheader("Soil Data Input")
            with st.form("soil_data_form"):
                ph = st.slider("Soil pH", 4.0, 8.5, 6.5, 0.1)
                nitrogen = st.number_input("Nitrogen (ppm)", 0.0, 100.0, 25.0)
                phosphorus = st.number_input("Phosphorus (ppm)", 0.0, 50.0, 15.0)
                potassium = st.number_input("Potassium (ppm)", 0.0, 500.0, 200.0)
                rainfall = st.number_input("Annual Rainfall (mm)", 0.0, 2000.0, 1000.0)
                temperature = st.number_input("Average Temperature (°C)", 0.0, 40.0, 22.0)
                crop = st.selectbox("Select Crop", list(self.analyzer.crop_requirements.keys()))
                
                submitted = st.form_submit_button("Analyze Soil")

        with col1:
            if submitted:
                soil_data = pd.DataFrame({
                    'pH': [ph],
                    'N': [nitrogen],
                    'P': [phosphorus],
                    'K': [potassium],
                    'Rainfall': [rainfall],
                    'Temperature': [temperature]
                })

                # Get predictions and recommendations
                predicted_yield = self.analyzer.predict_yield(soil_data)
                recommendations = self.analyzer.get_recommendations(soil_data, crop)

                # Display results
                st.subheader("Analysis Results")
                
                # Create metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Yield", f"{predicted_yield:.1f} kg/ha")
                with col2:
                    st.metric("Soil Health Score", f"{min(100, max(0, predicted_yield/50)):.1f}%")
                with col3:
                    st.metric("Optimization Potential", 
                             f"{max(0, min(100, (1000-predicted_yield)/10)):.1f}%")

                # Display recommendations
                st.subheader("Recommendations")
                for rec in recommendations:
                    if rec['type'] == 'warning':
                        st.warning(f"📊 {rec['message']}\n\n📋 Recommended Action: {rec['action']}")

                # Save to history
                history_entry = soil_data.copy()
                history_entry['Date'] = datetime.now()
                history_entry['Crop'] = crop
                history_entry['Predicted_Yield'] = predicted_yield
                st.session_state.soil_history = pd.concat([st.session_state.soil_history, history_entry])

    def historical_data_tab(self):
        st.subheader("Historical Soil Analysis Data")
        if not st.session_state.soil_history.empty:
            fig = px.line(st.session_state.soil_history, 
                         x='Date', 
                         y='Predicted_Yield',
                         title='Predicted Yield Over Time')
            st.plotly_chart(fig)
            
            st.dataframe(st.session_state.soil_history)
        else:
            st.info("No historical data available yet. Please perform some soil analyses first.")

    def reports_tab(self):
        st.subheader("Analysis Reports")
        if not st.session_state.soil_history.empty:
            report_type = st.selectbox("Select Report Type", 
                                     ["Soil Health Summary", "Nutrient Trends", "Yield Predictions"])
            
            if st.button("Generate Report"):
                st.subheader(f"{report_type} Report")
                self.generate_report(report_type)
        else:
            st.info("No data available for reporting. Please perform some soil analyses first.")

    def generate_report(self, report_type):
        if report_type == "Soil Health Summary":
            recent_data = st.session_state.soil_history.iloc[-1]
            st.write(f"Latest Soil Analysis Results ({recent_data['Date'].strftime('%Y-%m-%d')})")
            
            # Create gauge charts for each parameter
            fig = go.Figure()
            params = {'pH': (4, 8.5), 'N': (0, 100), 'P': (0, 50), 'K': (0, 500)}
            
            for param, (min_val, max_val) in params.items():
                fig.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = recent_data[param],
                    title = {'text': param},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [min_val, max_val]},
                        'steps': [
                            {'range': [min_val, (max_val-min_val)*0.3], 'color': "lightgray"},
                            {'range': [(max_val-min_val)*0.3, (max_val-min_val)*0.7], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': recent_data[param]
                        }
                    }
                ))
            
            st.plotly_chart(fig)

    def settings_tab(self):
        st.subheader("Application Settings")
        st.write("Configure your soil analysis preferences here.")
        
        # Example settings
        unit_system = st.selectbox("Unit System", ["Metric", "Imperial"])
        notification_preference = st.multiselect(
            "Notification Preferences",
            ["Email Alerts", "Mobile Notifications", "Weekly Reports"],
            ["Email Alerts"]
        )
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")

if __name__ == "__main__":
    app = SoilAnalysisApp()
    app.run()
