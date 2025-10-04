from data_processing.text_processing import text_clean, rem_stopwords, stem_text
import pandas as pd
import numpy as np


def reviews_clean(src_path, dest_path):
    """
    Cleans amazon's user reviews dataset.
    params:
    src_path : path for dataset
    dest_path : path where cleaned data will be stored
    """
    df = pd.read_json(src_path)
    features_not_req = ['reviewTime', 'style', 'image']
    df.drop(features_not_req, axis=1, inplace=True)
    
    to_impute = ['reviewerName', 'reviewText', 'summary']  # text features 
    df[to_impute] = df[to_impute].fillna('')
    df['vote'].fillna(0, inplace=True)  # most reviews are not voted by anyone.
    
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    # Nettoyage texte : nettoyage, suppression des stopwords, stemming
    for col in ['reviewText', 'summary']:
        df[col] = df[col].apply(text_clean)
        df[col] = df[col].apply(rem_stopwords)
        df[col] = df[col].apply(stem_text)
    
    df.to_json(dest_path, compression='gzip')
    return df


def meta_clean(src_path, dest_path):
    """
    Cleans amazon's user reviews meta dataset.
    params:
    src_path : path for dataset
    dest_path : path where cleaned data will be stored.
    """
    df = pd.read_json(src_path)
    features_not_req = ['category', 'tech1', 'fit', 'tech2', 'feature', 'date',
                        'image', 'main_cat', 'also_buy', 'rank', 'also_view',
                        'similar_item', 'details']
    # Suppression des colonnes non pertinentes ou très peu renseignées
    df.drop(features_not_req, axis=1, inplace=True)

    apply_func = lambda x: np.nan if (isinstance(x, list) and len(x) == 0) else (np.nan if x == '' else x)

    for col in df.columns:
        df[col] = df[col].apply(apply_func)  # remplacer listes vides et chaînes vides par NaN

    # Traitement des prix
    df['price'] = df['price'].apply(lambda x: str(x).strip(' $') if x is not None else np.nan)
    price_filter = lambda x: (float(x) if len(x) <= 6 else np.nan) if x is not None and not isinstance(x, float) else x
    df['price'] = df['price'].apply(price_filter).astype(float)

    # Description : concaténation si liste, sinon garder tel quel
    desc_filter = lambda x: (" ".join(x) if isinstance(x, list) else x) if x is not None else np.nan
    df['description'] = df['description'].apply(desc_filter)

    df.dropna(subset=['title'], inplace=True)  # suppression des lignes sans titre

    df['description'].fillna(df['title'], inplace=True)  # imputation des descriptions manquantes par le titre

    df['brand'].fillna("", inplace=True)  # imputation des marques manquantes par une chaîne vide

    # Nettoyage texte pour les colonnes textuelles
    df['title'] = df['title'].apply(text_clean)
    df['description'] = df['description'].apply(text_clean)

    df['price'].fillna(df['price'].mean(), inplace=True)  # imputation des prix manquants par la moyenne

    df['price'] = df['price'].apply(lambda x: round(x, 2))  # arrondi des prix à 2 décimales

    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)

    df.to_json(dest_path, compression='gzip')
    return df
