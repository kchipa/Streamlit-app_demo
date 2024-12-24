import streamlit as st
from forms.contact import contact_form

@st.dialog("Contact Me")
def show_contact_form():
    st.write("Email: k.chipashvili03@gmail.com")
    contact_form()


col1, col2 = st.columns(2, gap="small", vertical_alignment='center')
with col1:
    st.image("./assets/kostap.jpg", width=230)
with col2:
    st.title("Konstantine Chipashvili", anchor=False)
    st.write(
        "Junior Data Analyst"
    )
    if st.button("Contact Me"):
        show_contact_form()

# ---Skills ---
st.write("\n")
st.subheader("Hard Skills", anchor=False)
st.write(
    '''
    - Programming: Python (pandas, numpy, Scikit-learn) and SQL
    - Data Visualisation: Matplotlib, Seaborn, PowerBi, MS Excel
    - Databases: Postgres
    '''
)
