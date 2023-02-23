import streamlit as st

def get_speed_color(speed):
    if speed <= 35:
        return 'green'
    elif speed <= 55:
        return 'orange'
    else:
        return 'red'

st.title("🏃motorcycle speed calculator🏍️")



speed = st.number_input("Enter the speed of the vehicle (in km/h):")

if st.button("Calculate"):
    if speed >= 0:
        st.write(f"the speed is {speed} km/h.  you drive ")
        color = get_speed_color(speed)
        if color == 'green':
            status = 'Slow'
        elif color == 'orange':
            status = 'Fast'
        else:
            status = 'Fast a lot'
        st.markdown(f"<p style='color:{color}; font-size:36px;'>{status}!</p>", unsafe_allow_html=True)
    else:
        st.write("The car can't negative speed, okay!!.")
