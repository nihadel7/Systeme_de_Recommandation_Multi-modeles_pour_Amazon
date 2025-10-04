from data_processing import data_cleaning, data_merge, feature_genration
from models import lin_svc, nb
import pandas as pd

def run_all():
    rev_path = 'data/raw/All_Beauty_25.json.gz'
    rev_clean_path = 'data/processed/clean_reviews.json.gz'
    meta_path = 'data/raw/meta_All_Beauty_25.json.gz'
    meta_clean_path = 'data/processed/clean_meta.json.gz'

    # Nettoyage
    data_cleaning.reviews_clean(rev_path, rev_clean_path)
    data_cleaning.meta_clean(meta_path, meta_clean_path)
    
    # Entraînement
    lin_svc.train(rev_clean_path)
    nb.train(rev_clean_path)
    
    # Génération des features
    feature_genration.all_feature(rev_clean_path)
    
    # Fusion finale
    data_merge.final_data(dest_path='data/traitees/final.json.gz')

    # Sauvegarder un résumé ou indicateur que c'est prêt
    print("Prétraitement terminé et fichiers enregistrés.")

if __name__ == "__main__":
    run_all()
