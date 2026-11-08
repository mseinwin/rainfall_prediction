import streamlit as st
import pickle 
import pandas as pd

st.set_page_config(page_title='Should I Bring an Umbrella?')

df = pd.read_csv('weatherAUS.csv')
df.columns = df.columns.str.strip()

st.header("Should I Bring an Umbrella?")

def load_lr_model():
    with open('rainfall_logestic_model.pkl', 'rb') as f:
        return pickle.load(f)

def load_knn_model():  
    with open('rainfall_knn_model.pkl', 'rb') as f:
        return pickle.load(f)

with st.sidebar:
    st.image('images.jpg')
    st.text("Project: Rainfall Prediction")
    st.text("Student Name: Sein Win")
    st.text("Student ID: PIUS20240090")
    st.text("Class of 2028")

col1, col2, col3 = st.columns(3)

with col1:
    var1 = st.selectbox("Enter your location",sorted(df['Location'].dropna().unique()))
    
    var2 = st.number_input("Minimum temperature (in Celsius)", min_value = -10.0, max_value = 40.0, value = 15.0)
    
    var3 = st.number_input("Maximum temperature (in Celsius)", min_value = -5.0, max_value = 50.0, value = 25.0)

    var4 = st.number_input("Rainfall today (mm)", min_value = 0.0, max_value = 300.0, value = 0.0)

    var5 = st.number_input("Evaporation (mm)", min_value = 0.0, max_value = 100.0, value = 5.0)

    var6 = st.number_input("Sunshine (hours)", min_value = 0.0, max_value = 15.0, value = 8.0)

    var7 = st.selectbox ("Wind gust direction (m/h)", sorted(df['WindGustDir']. dropna().unique()))
with col2:
    var8 = st.number_input("Wind gust speed (km/h)", min_value = 0.0, max_value = 150.0, value = 35.0)

    var9 = st.selectbox ("Wind direction at 9 AM", sorted(df['WindDir9am']. dropna().unique()))

    var10 = st.selectbox ("Wind direction at 3 PM", sorted(df['WindDir3pm']. dropna().unique()))

    var11 = st.number_input ("Wind speed at 9 AM (km/hr)", min_value = 0.0, max_value = 100.0, value = 15.0)

    var12 = st.number_input ("Wind speed at 3 PM (km/hr)", min_value = 0.0, max_value = 100.0, value = 20.0)

    var13 = st.slider("Humidity at 9 AM (%)", min_value = 0.0, max_value = 100.0, value = 70.0)

    var14 = st.slider("Humidity at 3 PM (%)", min_value = 0.0, max_value = 100.0, value = 50.0)

with col3:

    var15 = st.number_input("Pressure at 9 AM (hPa)", min_value = 950.0, max_value = 1050.0, value = 1010.0)

    var16 = st.slider("Pressure at 3 PM (hPa)", min_value = 950.0, max_value = 1040.0, value = 1005.0)

    var17 = st.slider("Cloud at 9 AM (oktas)", min_value = 0, max_value = 8, value = 4)

    var18 = st.slider("Cloud at 3 PM (oktas)", min_value = 0, max_value = 8, value = 4)

    var19 = st.number_input("Temperature at 9 AM (in Celsius)", min_value = - 5.0, max_value = 45.0, value = 20.0)
    
    var20 = st.number_input("Temperature at 3 PM (in Celsius)", min_value = -5.0, max_value = 50.0, value = 25.0)

    var21 = st.selectbox ("Did it rain today?", ["No", "Yes"])

predict_button = st.button("Predict the Rain Tomorrow")

if predict_button:
    input_data = pd.DataFrame(
        [
            {
                "Location": var1,	
                "MinTemp":var2,	
                "MaxTemp":var3,	
                "Rainfall":var4,	
                "Evaporation":var5,	
                "Sunshine": var6,	
                "WindGustDir":var7,	
                "WindGustSpeed":var8,
                "WindDir9am":var9,
                "WindDir3pm":var10,
                "WindSpeed9am":var11,
                "WindSpeed3pm":var12,
                "Humidity9am":var13,
                "Humidity3pm":var14,
                "Pressure9am":var15,
                "Pressure3pm":var16,
                "Cloud9am":var17,
                "Cloud3pm":var18,
                "Temp9am":var19,
                "Temp3pm":var20,
                "RainToday":var21,
            }
        ]
    )
    
    #model = load_knn_model()
    model = load_lr_model()
    result = model.predict(input_data)[0]
    rain_probability = model.predict_proba(input_data)[0,1]

    st.subheader("Prediction Result")

    if result == "Yes":
        st.warning("Please bring an umbrella! Based on the weather information, the rain is predicted for tomorrow.")

    else:
        st.success("Based on the weather information, there will be no rain tomorrow! You probably don't need an umbrella.")

    st.write(f"Rain prediction: {result}")
    st.write(f"Probability of rain: {rain_probability *100:2f}%")

#with st.expander("Dataset"):
   # st.table(df)