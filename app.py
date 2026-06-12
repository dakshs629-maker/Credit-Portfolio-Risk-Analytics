import streamlit as st
import pandas as pd
import numpy as np
import pickle
import anthropic

# Load model and scaler
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

FEATURES = ['EXT_SOURCE_2', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
            'AMT_CREDIT', 'AGE', 'AMT_INCOME_TOTAL', 'DAYS_ID_PUBLISH', 'EXT_SOURCE_3',
            'AMT_GOODS_PRICE', 'DAYS_LAST_PHONE_CHANGE', 'REGION_POPULATION_RELATIVE',
            'ORGANIZATION_TYPE', 'NAME_INCOME_TYPE', 'REGION_RATING_CLIENT_W_CITY',
            'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REGION_RATING_CLIENT',
            'CODE_GENDER', 'FLAG_EMP_PHONE', 'REG_CITY_NOT_WORK_CITY',
            'FLAG_DOCUMENT_3', 'REG_CITY_NOT_LIVE_CITY', 'NAME_FAMILY_STATUS']

st.title('Credit Risk Scorecard')
st.subheader('Applicant Risk Assessment')

col1, col2 = st.columns(2)
with col1:
    ext_source_2 = st.slider('External Credit Score 2', 0.0, 1.0, 0.5)
    ext_source_3 = st.slider('External Credit Score 3', 0.0, 1.0, 0.5)
    age = st.number_input('Age', 18, 70, 35)
    income = st.number_input('Annual Income', 25000, 1000000, 150000)
with col2:
    amt_credit = st.number_input('Loan Amount', 45000, 4000000, 500000)
    amt_annuity = st.number_input('Annual Annuity', 1000, 260000, 25000)
    days_employed = st.number_input('Days Employed', 0, 20000, 2000)
    code_gender = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Female' if x==0 else 'Male')

if st.button('Assess Risk'):
    input_data = {f: 0 for f in FEATURES}
    input_data['EXT_SOURCE_2'] = ext_source_2
    input_data['EXT_SOURCE_3'] = ext_source_3
    input_data['AGE'] = age
    input_data['AMT_INCOME_TOTAL'] = np.log1p(income)
    input_data['AMT_CREDIT'] = np.log1p(amt_credit)
    input_data['AMT_ANNUITY'] = np.log1p(amt_annuity)
    input_data['DAYS_EMPLOYED'] = days_employed
    input_data['CODE_GENDER'] = code_gender

    df = pd.DataFrame([input_data])
    score = model.predict_proba(df)[0][1]

    if score >= 0.5:
        st.error(f'HIGH RISK — Default Probability: {score:.1%}')
    elif score >= 0.3:
        st.warning(f'MEDIUM RISK — Default Probability: {score:.1%}')
    else:
        st.success(f'LOW RISK — Default Probability: {score:.1%}')

    import os
    client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": f"""
        Credit risk assessment for applicant:
        - External Credit Score: {ext_source_2:.2f}
        - Age: {age}, Income: {income}, Loan: {amt_credit}
        - Days Employed: {days_employed}
        - Default Probability: {score:.1%}
        Provide risk level, top 3 factors, one recommendation. Under 100 words.
        """}]
    )
    st.write(message.content[0].text)
