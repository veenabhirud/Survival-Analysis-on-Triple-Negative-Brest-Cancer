import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


st.sidebar.header('User Input Features')

st.sidebar.markdown('tnbc Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        HPE = st.sidebar.selectbox('HPE',('AGCT','SCST','JGCT','Androgen Secreting Tumor'))
        Stage = st.sidebar.selectbox('Stage',('la','lc','lV','lla','llc','llla','lllc'))
        Tumor_Size = st.sidebar.selectbox('Tumor Size',('10 cm','9 cm','6 cm','15x10 cm','11x12 cm','9x9 cm','8x9','8.1x8.1 cm','7cm','12 cm','7x9 cm','20 cm','10.6x11.94 cm','8x8 cm','>10 cm','10x10','13x13','25 cm','18x20 cm','9x 10 cm'))
        Surgery = st.sidebar.selectbox('Surgery',('Complete Surgery','Incomplete Surgery','FSO','Optimal surgery'))
        surgerylevel = st.sidebar.selectbox('Surgery Level',('1','2','3'))
        Chemo_given_initially= st.sidebar.selectbox('Chemo given Initially',('nil','6 x CDDP +Ctx','6xEP','6xCDDP+ Ctx ; 6xCDDP + Etop','PEB 4 cycles','CDDP+Bleo +Vinb 3 cycles','6xCDDP+ VCR ; 6xCDDP +VP16 ;oral endoxan','BEP'))
        Age = st.sidebar.slider('Age', 14,67,40)
        
        data = {'Age': Age,
                'HPE': HPE,
                'Stage': Stage,
                'Tumor_Size': Tumor_Size,
                'Surgery': Surgery,
                'surgerylevel': surgerylevel,
                'Chemo_given_initially': Chemo_given_initially,}
        features = pd.DataFrame(data, index=[0])
        return features
    input_data = user_input_features()
    

# Combines user input features with entire dataset
# This will be useful for the encoding phase
data_raw = pd.read_csv('C:/Users/Hp/Downloads/TNBC_survival.xlsx - Sheet1.csv')
#Remove the column EmployeeNumber
data_raw = data_raw.drop(['Treatment given on relapse','Outcome_time','Survival ','event','relapse_time'], axis = 1)
data_raw=data_raw.fillna(0)
# A number assignment 

data = data_raw.drop(columns=['relapse'])
df = pd.concat([input_data,data],axis=0)

# Encoding of ordinal features
category_col =['HPE','Stage','Tumor_Size','Surgery','Chemo_given_initially']
for col in category_col:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('tnbc_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
tnbc_prediction = np.array(['Yes','No'])
st.write(tnbc_prediction[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)