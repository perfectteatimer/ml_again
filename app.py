import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


with open(Path(__file__).with_name("model.pickle"), "rb") as file:
    saved = pickle.load(file)

model = saved["model"]
st.set_page_config(page_title="цены на автомобили")
st.title("цены на автомобили")
forecast, data, weights = st.tabs(["прогноз", "данные", "веса модели"])

with forecast:
    car = {}
    with st.form("car"):
        left, right = st.columns(2)
        with left:
            car["year"] = st.number_input("год выпуска", 1900, pd.Timestamp.now().year, 2014)
            car["km_driven"] = st.number_input("пробег, км", min_value=0, value=70000, step=1000)
            car["engine"] = st.number_input("объем двигателя, куб. см", min_value=1, value=1248)
        with right:
            car["max_power"] = st.number_input("мощность, bhp", min_value=0.1, value=74.0)
            car["seats"] = st.selectbox("число мест", [2, 4, 5, 6, 7, 8, 9, 10, 14], index=2)
            car["mileage"] = st.number_input("км на литр или кг топлива", min_value=0.1, value=23.4)

        fuels = {"дизель": "Diesel", "бензин": "Petrol", "метан": "CNG", "пропан-бутан": "LPG"}
        sellers = {"владелец": "Individual", "дилер": "Dealer", "проверенный дилер": "Trustmark Dealer"}
        gearboxes = {"механика": "Manual", "автомат": "Automatic"}
        owners = {
            "первый": "First Owner",
            "второй": "Second Owner",
            "третий": "Third Owner",
            "четвертый или позже": "Fourth & Above Owner",
            "машина для тест-драйва": "Test Drive Car",
        }
        with left:
            fuel = st.selectbox("топливо", list(fuels))
            car["fuel"] = fuels[fuel]
            seller = st.selectbox("продавец", list(sellers))
            car["seller_type"] = sellers[seller]
        with right:
            gearbox = st.selectbox("коробка передач", list(gearboxes))
            car["transmission"] = gearboxes[gearbox]
            owner = st.selectbox("какой по счету владелец", list(owners))
            car["owner"] = owners[owner]

        submitted = st.form_submit_button("оценить цену")

    if submitted:
        features = pd.DataFrame([car])
        price = model.predict(features)[0]
        if price >= 1:
            price_text = f"{price:,.0f}".replace(",", " ")
            st.metric("прогноз цены", price_text + " ₹")
        else:
            st.warning("модель дала цену ниже одной рупии. для такой машины оценка ненадежна")

with data:
    st.subheader("медианная цена по году выпуска")
    st.line_chart(saved["price_by_year"], x_label="год выпуска", y_label="цена, рупии")
    st.subheader("медианная цена по типу коробки")
    prices = saved["price_by_transmission"].rename(index={"Manual": "механика", "Automatic": "автомат"})
    st.bar_chart(prices, x_label="коробка передач", y_label="цена, рупии")

with weights:
    st.subheader("коэффициенты модели")
    feature_names = model.named_steps["features"].get_feature_names_out()
    coefficients = pd.Series(model.named_steps["model"].coef_, index=feature_names)
    coefficients = coefficients.sort_values().rename("вес")
    st.bar_chart(coefficients, horizontal=True, sort=False, height=540, x_label="признак", y_label="вес")
