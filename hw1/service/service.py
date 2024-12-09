from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
import re
from io import StringIO


app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = make_dataframe([item])
    df = preprocess_items(df)
    return model.predict(df)

@app.post("/predict_items")
def predict_items(items=Body()):
    items_decoded = items.decode('utf-8')
    data = pd.read_csv(StringIO(items_decoded))
    df = preprocess_items(data)
    predictions = model.predict(df)
    df['selling_price'] = predictions

    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(output, media_type="text/csv", headers={
        "Content-Disposition": f"attachment; filename=output"
    })

def preprocess_items(df: pd.DataFrame) -> pd.DataFrame:
    df['mileage'] = df['mileage'].str.extract('(\d+\.?\d*)').astype(float)
    df['engine'] = df['engine'].str.extract('(\d+)').astype(int)
    df['max_power'] = df['max_power'].str.extract('(\d+\.?\d*)').astype(float)
    df['seats'] = df['seats'].astype(int)

    df[['torque', 'max_torque_rpm']] = pd.DataFrame(
        df['torque'].apply(split_torque).tolist(),
        index=df.index
    )
    df['name'] = df['name'].str.split(' ').str[0]
    df['power_per_l'] = df['max_power'] / (df['engine'] / 1000)
    df['year_squared'] = df['year'] ** 2

    numeric_columns = [
        'year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque',
        'max_torque_rpm', 'power_per_l', 'year_squared'
    ]

    category_cols = [
        'name', 'fuel', 'seller_type', 'transmission', 'owner'
    ]

    all_category_cols = [
        'is_official_dealer',
        'is_first_or_second_owner',
        'name_Ambassador',
        'name_Ashok',
        'name_Audi',
        'name_BMW',
        'name_Chevrolet',
        'name_Datsun',
        'name_Fiat',
        'name_Ford',
        'name_Honda',
        'name_Hyundai',
        'name_Jaguar',
        'name_Jeep',
        'name_Lexus',
        'name_Mahindra',
        'name_Maruti',
        'name_Mercedes-Benz',
        'name_Mitsubishi',
        'name_Nissan',
        'name_Opel',
        'name_Renault',
        'name_Skoda',
        'name_Tata',
        'name_Toyota',
        'name_Volkswagen',
        'name_Volvo',
        'fuel_CNG',
        'fuel_Diesel',
        'fuel_LPG',
        'fuel_Petrol',
        'seller_type_Dealer',
        'seller_type_Individual',
        'seller_type_Trustmark Dealer',
        'transmission_Automatic',
        'transmission_Manual',
        'owner_First Owner',
        'owner_Fourth & Above Owner',
        'owner_Second Owner',
        'owner_Test Drive Car',
        'owner_Third Owner'
    ]

    model_columns = [
        'fuel_Diesel',
        'owner_Second Owner',
        'name_Audi',
        'year_squared',
        'owner_First Owner',
        'max_power',
        'power_per_l',
        'mileage',
        'name_Honda',
        'owner_Test Drive Car',
        'seller_type_Dealer',
        'fuel_Petrol',
        'name_Renault',
        'name_Jeep',
        'name_Tata',
        'name_Lexus',
        'name_Ambassador',
        'is_first_or_second_owner',
        'name_Mitsubishi',
        'name_Toyota',
        'name_Mahindra',
        'seller_type_Individual',
        'name_Datsun',
        'name_Volvo',
        'transmission_Automatic',
        'max_torque_rpm',
        'engine',
        'seller_type_Trustmark Dealer',
        'seats',
        'torque',
        'name_Nissan',
        'name_Mercedes-Benz',
        'owner_Third Owner',
        'name_Skoda',
        'name_Fiat',
        'is_official_dealer',
        'name_BMW',
        'name_Maruti',
        'km_driven',
        'fuel_CNG',
        'name_Chevrolet',
        'name_Hyundai',
        'year',
        'owner_Fourth & Above Owner',
        'fuel_LPG',
        'name_Volkswagen',
        'name_Jaguar',
        'name_Ford',
        'transmission_Manual'
    ]

    df_numeric = scaler.transform(df[numeric_columns])
    df_numeric = pd.DataFrame(df_numeric, columns=numeric_columns)

    df_cat = pd.get_dummies(
        df[category_cols],
        columns=category_cols
    ).reindex(columns=all_category_cols, fill_value=0)

    df_cat['is_official_dealer'] = df['seller_type'].apply(lambda x: 1 if x == 'Dealer' else 0)
    df_cat['is_first_or_second_owner'] = df['owner'].apply(lambda x: 1 if x in ['First Owner', 'Second Owner'] else 0)
    df_cat['is_first_or_second_owner_and_official_dealer'] = df_cat['is_first_or_second_owner'] * df_cat['is_official_dealer']

    df_encoded = pd.concat([df_numeric.reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1)
    df_encoded = pd.concat([df_encoded.reset_index(drop=True), df['seats'].reset_index(drop=True)], axis=1)

    return df_encoded[model_columns]

def make_dataframe(items: List[Item]) -> pd.DataFrame:
    return pd.DataFrame([item.model_dump() for item in items])

def split_torque(torque_str):
    if pd.isna(torque_str):
        return pd.NA, pd.NA

    torque_str = torque_str.replace(',', '')
    torque_match = re.search(r'(\d+\.?\d*)', torque_str)
    rpm_match = re.search(r'@\s*(\d+\.?\d*)', torque_str)

    torque = float(torque_match.group(1)) if torque_match else pd.NA
    rpm = float(rpm_match.group(1)) if rpm_match else pd.NA

    if 'kgm' in torque_str.lower():
        torque = torque * 9.80665

    return torque, rpm
