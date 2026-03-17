"""
DatasetPreProcessing.py
────────────────────────
Pulizia e normalizzazione del dataset Airbnb Open Data.
Tutti i percorsi sono relativi alla root del progetto.
"""

import os
import numpy as npy
import pandas as pds
from datetime import datetime


class AirBnBDatasetPreprocessing:

    def __init__(self, input_file_path: str, output_file_path: str):
        self.original_file_path  = input_file_path
        self.processed_file_path = output_file_path
        self.data = None

    def load_data(self):
        try:
            self.data = pds.read_csv(self.original_file_path, low_memory=False)
            print(f"[PreProcessing] Caricamento completato: {self.data.shape}")
        except FileNotFoundError:
            print(f"[PreProcessing] File non trovato: {self.original_file_path}")
        except Exception as exc:
            print(f"[PreProcessing] Errore generico: {exc}")

    def clean_data(self):
        if self.data is None:
            print("[PreProcessing] Nessun dato da pulire.")
            return

        print("[PreProcessing] Inizio pulizia dati…")

        # Colonne non informative
        columns_to_drop = ['NAME', 'country', 'country code',
                           'license', 'host name', 'house_rules']
        self.data.drop(columns=[c for c in columns_to_drop if c in self.data.columns],
                       inplace=True)

        # Righe con valori mancanti critici
        self.data.dropna(subset=['neighbourhood group', 'neighbourhood',
                                 'lat', 'long', 'instant_bookable'], inplace=True)
        self.data.drop_duplicates(inplace=True)

        # ── Formattazione ────────────────────────────────────────────────
        for col in ['price', 'service fee']:
            if col in self.data.columns:
                self.data[col] = (self.data[col]
                                  .str.replace('$', '', regex=False)
                                  .str.replace(',', '', regex=False)
                                  .str.replace(' ', '', regex=False))
                self.data[col] = pds.to_numeric(self.data[col], errors='coerce')

        if 'Construction year' in self.data.columns:
            self.data['Construction year'] = pds.to_numeric(
                self.data['Construction year'], errors='coerce').astype('Int64')
        if 'minimum nights' in self.data.columns:
            self.data['minimum nights'] = pds.to_numeric(
                self.data['minimum nights'], errors='coerce').astype('Int64')

        # Date
        self.data['last review'] = pds.to_datetime(self.data['last review'], errors='coerce')
        mean_date = self.data['last review'].mean()
        self.data['last review'] = self.data['last review'].fillna(mean_date)
        self.data['last review'] = pds.to_datetime(self.data['last review'])

        self.data['room type'] = self.data['room type'].astype(str)

        # availability 365: porta in [0, 365]
        self.data['availability 365'] = npy.where(
            self.data['availability 365'] < 0,
            self.data['availability 365'] * -1,
            self.data['availability 365']
        )
        self.data['availability 365'] = npy.where(
            self.data['availability 365'] > 365, 365,
            self.data['availability 365']
        )

        # minimum nights: porta in [1, 365]
        self.data['minimum nights'] = (self.data['minimum nights']
                                       .mask(self.data['minimum nights'] < 0,
                                             self.data['minimum nights'] * -1)
                                       .mask(self.data['minimum nights'] > 365, 365)
                                       .fillna(1))

        # Typo neighbourhood group
        self.data['neighbourhood group'] = (
            self.data['neighbourhood group'].replace('brookln', 'Brooklyn')
        )

        # ── Completamento ────────────────────────────────────────────────
        self.data['price'] = self.data['price'].fillna(
            self.data['price'].mean().astype(int))
        self.data['service fee'] = self.data['service fee'].fillna(
            self.data['service fee'].mean())

        self.data['host_identity_verified'] = (
            self.data['host_identity_verified'].fillna('unconfirmed'))

        self.data['calculated host listings count'] = (
            self.data.groupby('host id')['calculated host listings count']
            .transform(lambda x: x.fillna(x.count()))
        )

        modal_cp = self.data['cancellation_policy'].mode()[0]
        self.data['cancellation_policy'] = (
            self.data['cancellation_policy'].fillna(modal_cp))

        cy_mean = self.data.groupby('neighbourhood group')['Construction year'].mean()
        self.data['Construction year'] = self.data.apply(
            lambda row: cy_mean[row['neighbourhood group']]
            if pds.isnull(row['Construction year']) else row['Construction year'],
            axis=1
        ).astype(int)

        for col, agg in [('number of reviews', 'median'),
                         ('reviews per month', 'median'),
                         ('availability 365', 'median'),
                         ('review rate number', 'mean')]:
            if col in self.data.columns:
                fill_val = (getattr(self.data[col], agg)()
                            if agg == 'median' else self.data[col].mean())
                self.data[col] = self.data[col].fillna(fill_val)

        self.data.loc[self.data['number of reviews'] == 0, 'reviews per month'] = 0

        # Date future → oggi
        today = pds.Timestamp.today().normalize()
        self.data['last review'] = self.data['last review'].apply(
            lambda x: today if pds.notna(x) and x > today else x)
        self.data['last review'] = pds.to_datetime(self.data['last review'])
        self.data['last review'] = self.data['last review'].fillna(
            self.data['last review'].mean())
        self.data['last review'] = pds.to_datetime(self.data['last review'])

        print(f"[PreProcessing] Pulizia completata. Shape finale: {self.data.shape}")

    def save_processed_data(self):
        output_path = os.path.join(self.processed_file_path, 'Post_PreProcessing')
        os.makedirs(output_path, exist_ok=True)
        if self.data is not None:
            out_file = os.path.join(output_path, 'Airbnb_Processed_Data.csv')
            self.data.to_csv(out_file, index=False)
            print(f"[PreProcessing] Dati salvati in '{out_file}'.")

    def show_data(self):
        if self.data is not None:
            self.data.info()


def call():
    preprocessor = AirBnBDatasetPreprocessing(
        input_file_path=os.path.join('data', 'Airbnb_Open_Data.csv'),
        output_file_path='data'
    )
    preprocessor.load_data()
    preprocessor.show_data()
    preprocessor.clean_data()
    preprocessor.show_data()
    preprocessor.save_processed_data()
