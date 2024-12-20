import streamlit as st 

# --- Page Setup ---
about_page = st.Page(
    page='views/about_me.py',
    title="About Me",
    icon =":material/account_circle:",
    default=True,
)
crypto_stocks_prediction_app = st.Page(
    page='views/crypto_stocks_prediction_app.py',
    title="Crypto Stocks Prediction App",
    icon = ":material/linked_services:",
)
#movie_recommendation_app = st.Page(
 #   page='views/crypto_stocks_prediction_app.py',
  #  title="Crypto Stocks Prediction App",
   # icon = ":material/linked_services:",
#)

#pg = st.navigation(pages=[about_page, crypto_stocks_prediction_app])
pg = st.navigation(
    {
        "Info": [about_page],
        "Projects": [crypto_stocks_prediction_app],
    }
)

st.sidebar.text("Made by Kosta")
pg.run()
