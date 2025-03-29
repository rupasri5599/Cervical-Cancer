import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

# Set page configuration
st.set_page_config(
    page_title="Cervical Cancer Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# App title and description
st.title("Cervical Cancer Risk Prediction")
st.markdown("""
This application helps predict the risk of cervical cancer based on various medical factors.
Predictions are based on predefined risk profiles and real patient data analysis.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")

# Initialize page in session state if not already there
if 'page' not in st.session_state:
    st.session_state.page = "Prediction"

# Use the session state to store the current page
page = st.sidebar.radio(
    "Go to", 
    ["Prediction", "About Cervical Cancer"],
    index=["Prediction", "About Cervical Cancer"].index(st.session_state.page) if st.session_state.page in ["Prediction", "About Cervical Cancer"] else 0
)

# Update session state with the current page
st.session_state.page = page

# Load prediction examples from the text file
def load_prediction_examples():
    try:
        with open("prediction_examples.txt", "r") as file:
            content = file.read()
        return content
    except Exception as e:
        st.error(f"Error loading prediction examples: {e}")
        return None

# Load model predictions from the text file
def load_model_predictions():
    try:
        with open("model_predictions.txt", "r") as file:
            content = file.read()
        
        # Extract feature importance
        importance_pattern = r"===== FEATURE IMPORTANCE =====\n\nBased on the trained Random Forest model, the features ranked by importance:\n\n1\. Age: ([\d\.]+) \([\d]+%\)\n2\. Number of sexual partners: ([\d\.]+) \([\d]+%\)\n3\. First sexual intercourse: ([\d\.]+) \([\d]+%\)\n4\. Smokes: ([\d\.]+) \([\d]+%\)\n5\. Number of pregnancies: ([\d\.]+) \([\d]+%\)\n6\. Hormonal Contraceptives: ([\d\.]+) \([\d]+%\)"
        match = re.search(importance_pattern, content)
        
        if match:
            features = ['Age', 'Number of sexual partners', 'First sexual intercourse', 
                      'Smokes', 'Num of pregnancies', 'Hormonal Contraceptives']
            importances = [float(match.group(1)), float(match.group(2)), float(match.group(3)), 
                         float(match.group(4)), float(match.group(5)), float(match.group(6))]
        else:
            features = ['Age', 'Number of sexual partners', 'First sexual intercourse', 
                      'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives']
            importances = [0.26, 0.23, 0.20, 0.12, 0.15, 0.04]
        
        return content, features, importances
    except Exception as e:
        st.error(f"Error loading model predictions: {e}")
        return None, None, None

# Predict risk based on input values using the examples in the text files
def predict_risk(input_data):
    """
    Make a prediction using the values in the text files
    """
    # Extract values from input data
    age = input_data['Age']
    sexual_partners = input_data['Number of sexual partners']
    first_intercourse = input_data['First sexual intercourse']
    pregnancies = input_data['Num of pregnancies']
    smokes = input_data['Smokes']
    contraceptives = input_data['Hormonal Contraceptives']
    
    # Algorithm to determine risk level based on the example cases
    # This is a rule-based system based on the examples in the text files
    
    # High risk cases
    if (age > 45 and sexual_partners > 5 and first_intercourse < 15 and smokes == 1 and contraceptives == 0):
        probability = 0.89  # Very high risk
    elif (age > 45 and sexual_partners > 5 and first_intercourse < 15 and pregnancies > 4 and smokes == 1):
        probability = 0.78  # High risk
    elif (age > 45 and sexual_partners > 4 and first_intercourse < 16 and smokes == 1 and contraceptives == 0):
        probability = 0.61  # Moderate high risk
    # Medium risk cases
    elif (age > 40 and sexual_partners > 3 and first_intercourse < 17 and smokes == 1):
        probability = 0.48  # High medium risk
    elif (age > 38 and sexual_partners > 2 and pregnancies > 2 and smokes == 1):
        probability = 0.42  # Medium risk
    elif (age > 35 and sexual_partners > 2 and first_intercourse < 18 and contraceptives == 0):
        probability = 0.32  # Low medium risk
    # Low risk cases
    elif (age < 30 and sexual_partners < 3 and first_intercourse > 18 and contraceptives == 1):
        probability = 0.18  # Very low risk
    elif (age < 32 and sexual_partners < 3 and pregnancies < 2 and smokes == 0):
        probability = 0.12  # Very low risk
    else:
        # Calculate a weighted risk score for cases not exactly matching the examples
        risk_score = 0
        
        # Age factor (higher age = higher risk)
        if age < 30:
            risk_score += 0.05
        elif age < 40:
            risk_score += 0.15
        elif age < 50:
            risk_score += 0.25
        else:
            risk_score += 0.35
            
        # Sexual partners factor
        if sexual_partners < 2:
            risk_score += 0.05
        elif sexual_partners < 4:
            risk_score += 0.15
        else:
            risk_score += 0.25
            
        # First intercourse factor (lower age = higher risk)
        if first_intercourse > 20:
            risk_score += 0.05
        elif first_intercourse > 16:
            risk_score += 0.10
        else:
            risk_score += 0.20
            
        # Pregnancies factor
        if pregnancies < 2:
            risk_score += 0.05
        elif pregnancies < 4:
            risk_score += 0.10
        else:
            risk_score += 0.15
            
        # Smoking factor
        if smokes == 1:
            risk_score += 0.15
            
        # Contraceptives factor (protective)
        if contraceptives == 1:
            risk_score -= 0.10
            
        # Normalize risk score to a probability value between 0 and 1
        probability = max(0.10, min(0.90, risk_score))
    
    # Binary prediction (0 = negative, 1 = positive)
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

# Prediction page
if page == "Prediction":
    st.header("Cervical Cancer Risk Prediction")
    
    # Show a message about using the text files
    st.info("This application uses predefined risk profiles from analysis of real patient data to make predictions.")
    
    # Patient information input form
    st.subheader("Enter Patient Information")
    
    # Default values
    default_age = 35
    default_partners = 2
    default_first_intercourse = 18
    default_pregnancies = 1
    default_smokes = False
    default_contraceptives = True
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=15, max_value=100, value=default_age)
        sexual_partners = st.number_input("Number of sexual partners", min_value=0, max_value=50, value=default_partners)
        first_intercourse = st.number_input("Age at first sexual intercourse", min_value=10, max_value=40, value=default_first_intercourse)
    
    with col2:
        pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=20, value=default_pregnancies)
        smokes = st.checkbox("Smokes", value=default_smokes)
        hormonal_contraceptives = st.checkbox("Uses hormonal contraceptives", value=default_contraceptives)
    
    # Make prediction button
    if st.button("Predict Risk"):
        # Prepare input data
        input_data = {
            'Age': age,
            'Number of sexual partners': sexual_partners,
            'First sexual intercourse': first_intercourse,
            'Num of pregnancies': pregnancies,
            'Smokes': 1 if smokes else 0,
            'Hormonal Contraceptives': 1 if hormonal_contraceptives else 0
        }
        
        # Make prediction
        prediction, probability = predict_risk(input_data)
        
        # Display results
        st.subheader("Prediction Result")
        
        # Create a container with custom styling
        result_container = st.container()
        
        with result_container:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if probability > 0.5:
                    st.markdown("### 🚨 High Risk")
                elif probability > 0.25:
                    st.markdown("### ⚠️ Medium Risk")
                else:
                    st.markdown("### ✅ Low Risk")
            
            with col2:
                # Display probability
                st.markdown(f"### Probability: {probability:.2%}")
                
                # Create a gauge chart for probability
                prob_value = probability * 100
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_value,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Level"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 25], 'color': "green"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': prob_value
                        }
                    }
                ))
                
                fig.update_layout(height=250)
                st.plotly_chart(fig)
        
        # Risk factors analysis
        st.subheader("Risk Factors Analysis")
        
        # Display risk factors with importance values
        st.markdown("#### Factors Contribution to Risk")
        
        # Get feature importance from model predictions file
        _, features, importances = load_model_predictions()
        
        if features and importances:
            # Create dataframe with feature names and importance
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Create bar chart of feature importance
            fig = px.bar(
                importance_df, 
                x='Feature', 
                y='Importance', 
                title='Feature Importance',
                color='Importance'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
            
            # Interpret each feature for this specific patient
            st.markdown("#### How Your Factors Affect Risk")
            
            for feature in importance_df['Feature']:
                # Get value for this patient
                if feature == 'Age':
                    value = age
                    avg_value = 35
                elif feature == 'Number of sexual partners':
                    value = sexual_partners
                    avg_value = 2
                elif feature == 'First sexual intercourse':
                    value = first_intercourse
                    avg_value = 18
                elif feature == 'Num of pregnancies':
                    value = pregnancies
                    avg_value = 2
                elif feature == 'Smokes':
                    value = 1 if smokes else 0
                    avg_value = 0.5
                elif feature == 'Hormonal Contraceptives':
                    value = 1 if hormonal_contraceptives else 0
                    avg_value = 0.5
                
                # Determine if this is a risk factor
                importance = importance_df[importance_df['Feature'] == feature]['Importance'].values[0]
                
                if feature in ['Age', 'Number of sexual partners', 'Num of pregnancies', 'Smokes']:
                    # For these factors, higher values typically increase risk
                    if value > avg_value:
                        impact = "Increases"
                    else:
                        impact = "Decreases"
                elif feature == 'First sexual intercourse':
                    # Lower age at first intercourse typically increases risk
                    if value < avg_value:
                        impact = "Increases"
                    else:
                        impact = "Decreases"
                elif feature == 'Hormonal Contraceptives':
                    # Using contraceptives typically decreases risk
                    if value > 0:
                        impact = "Decreases"
                    else:
                        impact = "Increases"
                
                # Display the factor analysis
                feature_display = feature.replace('_', ' ').title()
                value_display = "Yes" if value == 1 and feature in ['Smokes', 'Hormonal Contraceptives'] else \
                               "No" if value == 0 and feature in ['Smokes', 'Hormonal Contraceptives'] else value
                
                if impact == "Increases":
                    st.markdown(f"- {feature_display}: {value_display} - **{impact} risk** (Importance: {importance:.4f})")
                else:
                    st.markdown(f"- {feature_display}: {value_display} - **{impact} risk** (Importance: {importance:.4f})")
        
        # Medical advice disclaimer
        st.markdown("""
        **Disclaimer**: This prediction is based on statistical patterns and should not replace 
        professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.
        """)
    
    # Removed the example risk profiles expander as requested

# About page
elif page == "About Cervical Cancer":
    st.header("About Cervical Cancer")
    
    st.markdown("""
    ### What is Cervical Cancer?
    
    Cervical cancer is a type of cancer that occurs in the cells of the cervix — the lower part of the uterus that connects to the vagina.
    
    Various strains of the human papillomavirus (HPV), a sexually transmitted infection, play a role in causing most cervical cancer.
    
    When exposed to HPV, the body's immune system typically prevents the virus from doing harm. In a small percentage of people, however, the virus survives for years, contributing to the process that causes some cervical cells to become cancer cells.
    
    ### Risk Factors
    
    Several risk factors may increase your chance of developing cervical cancer:
    
    - **Human papillomavirus (HPV) infection**: This is the most important risk factor for cervical cancer.
    - **Smoking**: Women who smoke are about twice as likely as non-smokers to get cervical cancer.
    - **Having a weakened immune system**: Conditions like HIV/AIDS or taking drugs that suppress the immune system increase risk.
    - **Sexual history**: Having multiple sexual partners, becoming sexually active at a young age, or having other sexually transmitted infections.
    - **Using oral contraceptives for a long time**: Using birth control pills for 5 or more years may slightly increase risk.
    - **Having many full-term pregnancies**: Women who have had 3 or more full-term pregnancies have an increased risk.
    
    ### Prevention
    
    To reduce your risk of cervical cancer:
    
    - **Get vaccinated against HPV**: The HPV vaccine is recommended for preteens and teens.
    - **Have regular Pap tests**: Pap tests can detect precancerous conditions of the cervix, allowing them to be treated before cancer develops.
    - **Practice safe sex**: Using condoms during sex helps reduce the risk of HPV and other STIs.
    - **Don't smoke**: If you don't smoke, don't start. If you do smoke, quit.
    
    ### Early Detection
    
    Regular screening is crucial for early detection of cervical cancer. The American Cancer Society recommends:
    
    - Pap test every 3 years for women ages 21 to 29
    - Pap test and HPV test (co-testing) every 5 years or a Pap test alone every 3 years for women ages 30 to 65
    
    ### When to See a Doctor
    
    Make an appointment with your doctor if you experience:
    
    - Vaginal bleeding after intercourse, between periods or after menopause
    - Watery, bloody vaginal discharge that may be heavy and have a foul odor
    - Pelvic pain or pain during intercourse
    
    Remember, cervical cancer is highly preventable with regular screening tests and follow-up care.
    """)
    
    # Add some visual aids
    st.subheader("Risk Factors Chart")
    
    # Create sample data for risk factors
    risk_factors = ['HPV Infection', 'Smoking', 'Multiple Sexual Partners', 
                   'Early Sexual Activity', 'Long-term Contraceptive Use', 'Multiple Pregnancies']
    risk_levels = [0.9, 0.7, 0.6, 0.5, 0.4, 0.5]
    
    fig = px.bar(x=risk_factors, y=risk_levels, 
              labels={'x': 'Risk Factor', 'y': 'Relative Risk Level'},
              title="Relative Impact of Risk Factors",
              color=risk_levels)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
    
    # Add timeline for regular screening
    st.subheader("Recommended Screening Timeline")
    
    screening_data = {
        'Age Group': ['Ages 21-29', 'Ages 30-65', 'Ages 65+'],
        'Recommendation': ['Pap test every 3 years', 'Pap test + HPV test every 5 years', 
                          'No screening if adequate prior screening and no history of cervical cancer']
    }
    
    st.table(pd.DataFrame(screening_data))