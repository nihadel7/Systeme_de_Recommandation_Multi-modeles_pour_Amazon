import streamlit as st
import pandas as pd
import json
from recommendation_filters import content_based_filter, popularity_filter
from models import collaborative_model_based
import time

# Configuration de la page
st.set_page_config(
    page_title="Système de Recommandation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .default-model-card {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid #28a745;
    }
    
    .recommendation-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .product-image {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        max-width: 120px;
        max-height: 120px;
        object-fit: cover;
    }
    
    .product-info {
        flex: 1;
    }
    
    .product-title {
        margin: 0 0 0.5rem 0;
        color: #333;
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    .product-details {
        color: #666;
        font-size: 0.9rem;
        margin: 0.2rem 0;
    }
    
    .price-tag {
        background: #28a745;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .no-image-placeholder {
        width: 120px;
        height: 120px;
        background: linear-gradient(135deg, #e0e0e0 0%, #bdbdbd 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #666;
        font-size: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    
    .default-badge {
        background: #28a745;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 8px;
    }
    
    .rating-stars {
        color: #ffc107;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True) 

# Fonction de chargement des métadonnées avec images
@st.cache_data(show_spinner=False)
def load_metadata():
    """Charge les métadonnées des produits incluant les images"""
    try:
        metadata = {}
        with open('data/meta_All_Beauty[1].jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    # Extraire les informations importantes
                    asin = data.get('parent_asin', '')
                    if not asin:
                        continue
                    
                    # Extraire la meilleure image disponible
                    image_url = None
                    if data.get('images'):
                        for img in data['images']:
                            if img.get('variant') == 'MAIN':
                                image_url = img.get('large') or img.get('thumb')
                                break
                        if not image_url and data['images']:
                            # Prendre la première image si pas d'image principale
                            image_url = data['images'][0].get('large') or data['images'][0].get('thumb')
                    
                    metadata[asin] = {
                        'title': data.get('title', ''),
                        'image_url': image_url,
                        'average_rating': data.get('average_rating', 0),
                        'rating_number': data.get('rating_number', 0),
                        'price': data.get('price', None),
                        'store': data.get('store', ''),
                        'main_category': data.get('main_category', ''),
                        'features': data.get('features', [])
                    }
                except json.JSONDecodeError:
                    continue
        
        return metadata
    except FileNotFoundError:
        st.warning("⚠️ Fichier de métadonnées non trouvé. Les images ne seront pas affichées.")
        return {}
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des métadonnées: {str(e)}")
        return {}

# Fonction de chargement des données avec cache
@st.cache_data(show_spinner=False)
def load_data():
    with st.spinner("🔄 Chargement des données..."):
        try:
            name_df = pd.read_json('data/asin_title.json.gz')
            final_df = pd.read_json('data/traitees/final.json.gz')
            df = content_based_filter.cbf_data('data/traitees/final.json.gz')
            idx = content_based_filter.indices(df)
            cosim = content_based_filter.cosine_sim(df['description'])
            svd_model = collaborative_model_based.train(
                df_path='data/traitees/final.json.gz', 
                sample_frac=0.5, 
                idx='asin',
                col='reviewerID', 
                val='positive_prob'
            )
            metadata = load_metadata()
            return name_df, final_df, df, idx, cosim, svd_model, metadata
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des données: {str(e)}")
            return None, None, None, None, None, None, {}

def display_product_card(row, metadata, index):
    """Affiche une carte produit avec image et informations"""
    asin = row['asin']
    title = row.get('title', 'Titre non disponible')
    
    # Récupérer les métadonnées
    meta_info = metadata.get(asin, {})
    image_url = meta_info.get('image_url')
    rating = meta_info.get('average_rating', 0)
    rating_count = meta_info.get('rating_number', 0)
    price = meta_info.get('price')
    store = meta_info.get('store', '')
    
    # Générer les étoiles
    stars = '⭐' * int(rating) if rating > 0 else ''
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if image_url:
            try:
                st.image(image_url, width=120, caption="")
            except:
                st.markdown("""
                <div class="no-image-placeholder">
                    🖼️
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="no-image-placeholder">
                🖼️
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="product-info">
            <h4 class="product-title">{title[:100]}{'...' if len(title) > 100 else ''}</h4>
            <div class="product-details">
                <strong>ASIN:</strong> {asin}
            </div>
            <div class="product-details">
                <strong>Magasin:</strong> {store if store else 'Non spécifié'}
            </div>
            <div class="rating-stars">
                {stars} {f'({rating}/5)' if rating > 0 else 'Non noté'} 
                {f'• {rating_count} avis' if rating_count > 0 else ''}
            </div>
            {f'<div class="price-tag">{price}</div>' if price else ''}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

# Interface principale
def main():
    # En-tête
    st.markdown("<h1 style='display: inline; vertical-align: middle;'>Système de Recommandation Intelligent</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Découvrez des produits parfaitement adaptés à vos goûts</p>', unsafe_allow_html=True)
    
    # Chargement des données
    data_load_state = st.text('🔄 Chargement des données...')
    name_df, final_df, df, idx, cosim, svd_model, metadata = load_data()
    
    if name_df is None:
        st.error("❌ Impossible de charger les données. Vérifiez vos fichiers.")
        return
    
    data_load_state.text('✅ Données chargées avec succès!')
    time.sleep(0.5)
    data_load_state.empty()
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.image("Amazon.png", width=110)

        st.markdown("## Paramètres")
        
        # Informations sur les modèles
        with st.expander("ℹ️ À propos des modèles"):
            st.markdown("""
            **Basé contenu**  *(Recommandé)*: Analyse la description des produits pour trouver des articles similaires. Idéal pour découvrir des produits avec des caractéristiques semblables.
            
            **Popularité**: Recommande les produits les plus populaires et les mieux notés du catalogue.
            
            **Collaboratif**: Utilise les préférences d'utilisateurs ayant des goûts similaires pour faire des recommandations personnalisées.
            
            **Hybride**: Combine plusieurs approches (collaboratif + contenu + popularité) pour des résultats optimaux.
            """)
        
        # Statistiques
        st.markdown("### 📊 Statistiques")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Produits", "3,917")  # Valeur fixe
        with col2:
            st.metric("Données", f"{len(final_df):,}")
    
    # Interface principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sélection du modèle avec des descriptions
        st.markdown("### Choisir votre modèle de recommandation")
        
        # Définition des modèles avec le modèle par défaut
        model_options = [
            "Basé contenu",  # Modèle par défaut en première position
            "Popularité", 
            "Collaboratif", 
            "Hybride"
        ]
        
        model_descriptions = {
            "Basé contenu": "📝 Produits similaires par contenu",
            "Popularité": "🔥 Produits tendance et populaires",
            "Collaboratif": "👥 Basé sur les utilisateurs similaires",
            "Hybride": "🔄 Combinaison intelligente"
        }
        
        # Fonction pour formater les options avec badge par défaut
        def format_model_option(model):
            if model == "Basé contenu":
                return f"{model_descriptions[model]} ⭐ (Recommandé)"
            return model_descriptions[model]
        
        model_choice = st.selectbox(
            "Modèle:",
            options=model_options,
            index=0,  # Index 0 = "Basé contenu" par défaut
            format_func=format_model_option,
            help="Le modèle 'Basé contenu' est recommandé pour des résultats précis basés sur les caractéristiques des produits"
        )
        
        # Message informatif pour le modèle par défaut
        if model_choice == "Basé contenu":
            st.info("✨ **Modèle recommandé sélectionné** : Ce modèle analyse les descriptions des produits pour vous proposer des articles aux caractéristiques similaires.")
        
        # Sélection du produit
        st.markdown("### 🛒 Sélectionner un produit")
        
        # Barre de recherche
        search_term = st.text_input(
            "🔍 Rechercher un produit:",
            placeholder="Tapez le nom du produit...",
            help="Commencez à taper pour filtrer les produits"
        )
        
        # Filtrer les produits selon la recherche
        filtered_products = name_df['title'].dropna().unique()
        if search_term:
            filtered_products = [p for p in filtered_products if search_term.lower() in p.lower()]
        
        if len(filtered_products) == 0:
            st.warning("⚠️ Aucun produit trouvé avec ce terme de recherche.")
            return
        
        # Limiter à 100 résultats pour la performance
        filtered_products = filtered_products[:100]
        
        product_title = st.selectbox(
            "Produit:",
            options=filtered_products,
            help=f"{'Résultats filtrés' if search_term else 'Tous les produits'} ({len(filtered_products)} éléments)"
        )
    
    with col2:
        st.markdown("###  Modèle sélectionné")
        
        # Affichage différent pour le modèle par défaut
        if model_choice == "Basé contenu":
            st.markdown(f'''
            <div class="default-model-card">
                <h4>{model_choice} <span class="default-badge">RECOMMANDÉ</span></h4>
                <p>{model_descriptions[model_choice]}</p>
                <small>✨ Modèle optimal pour des recommandations précises</small>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="model-card">
                <h4>{model_choice}</h4>
                <p>{model_descriptions[model_choice]}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        if product_title:
            st.markdown("### 📦 Produit choisi")
            # Afficher l'image du produit sélectionné si disponible
            product_row = name_df[name_df['title'] == product_title]
            if not product_row.empty:
                product_asin = product_row['asin'].values[0]
                meta_info = metadata.get(product_asin, {})
                image_url = meta_info.get('image_url')
                
                if image_url:
                    try:
                        st.image(image_url, width=150, caption="Produit sélectionné")
                    except:
                        st.info(f"**{product_title[:100]}**{'...' if len(product_title) > 100 else ''}")
                else:
                    st.info(f"**{product_title[:100]}**{'...' if len(product_title) > 100 else ''}")
            else:
                st.info(f"**{product_title[:100]}**{'...' if len(product_title) > 100 else ''}")
    
    # Bouton de recommandation centré
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        recommend_button = st.button(
            f"🚀 Générer les recommandations ({model_choice})",
            use_container_width=True,
            type="primary"
        )
    
    # Génération des recommandations
    if recommend_button:
        if not product_title:
            st.warning("⚠️ Veuillez sélectionner un produit.")
            return
        
        # Trouver l'ASIN correspondant
        product_row = name_df[name_df['title'] == product_title]
        product_asin = product_row['asin'].values[0] if not product_row.empty else None
        
        if not product_asin:
            st.error("❌ Produit non trouvé dans la base de données.")
            return
        
        # Générer les recommandations avec indicateur de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text(f"🔄 Génération des recommandations avec le modèle {model_choice}...")
            progress_bar.progress(25)
            
            if model_choice == "Basé contenu":
                recs = content_based_filter.recommend(
                    prod_asin=product_asin, cosine_sim=cosim, indices=idx,
                    cbf_df=df, lim=5, min_rate=2
                )
            elif model_choice == "Popularité":
                recs = popularity_filter.recommend(
                    df_path='data/traitees/final.json.gz',
                    rev_count=25, rating=3, sentiment=0.6
                )
            elif model_choice == "Collaboratif":
                recs = collaborative_model_based.recommend(
                    product=product_asin, model=svd_model, corr_thresh=0.5
                )
            else:  # Hybride
                recs = collaborative_model_based.recommend(
                    product=product_asin, model=svd_model, corr_thresh=0.5
                )
                if len(recs) < 5:
                    recs.extend(content_based_filter.recommend(
                        prod_asin=product_asin, cosine_sim=cosim, indices=idx,
                        cbf_df=df, lim=5, min_rate=2
                    ))
                if len(recs) < 5:
                    recs.extend(popularity_filter.recommend(
                        df_path='data/traitees/final.json.gz',
                        rev_count=25, rating=3, sentiment=0.6
                    ))
            
            progress_bar.progress(75)
            status_text.text("📊 Traitement des résultats...")
            
            # Traiter les recommandations
            if not recs:
                st.warning("⚠️ Aucune recommandation trouvée pour ce produit.")
                return
            
            recs_df = pd.DataFrame(recs[:10], columns=['asin']).merge(
                name_df, on='asin', how='left'
            )
            
            progress_bar.progress(100)
            status_text.text("✅ Recommandations générées!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # Affichage des résultats
            st.markdown("---")
            st.markdown("## 🎉 Vos Recommandations Personnalisées")
            
            if len(recs_df) > 0:
                # Métriques
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📦 Produits recommandés", len(recs_df))
                with col2:
                    st.metric("Modèle utilisé", model_choice)
                with col3:
                    quality_score = "Excellente" if model_choice == "Basé contenu" else "Élevée"
                    st.metric("⭐ Pertinence", quality_score)
                with col4:
                    images_count = sum(1 for _, row in recs_df.iterrows() if metadata.get(row['asin'], {}).get('image_url'))
                    st.metric("🖼️ Avec images", f"{images_count}/{len(recs_df)}")
                
                st.markdown("### 📋 Liste des recommandations")
                
                # Message spécial pour le modèle par défaut
                if model_choice == "Basé contenu":
                    st.success("✨ **Recommandations basées sur le contenu** : Ces produits ont été sélectionnés en analysant leurs descriptions et caractéristiques similaires au produit choisi.")
                
                # Affichage en cartes avec images
                for i, (_, row) in enumerate(recs_df.iterrows()):
                    with st.container():
                        display_product_card(row, metadata, i)
                
                # Tableau détaillé
                with st.expander("📊 Vue tableau détaillée"):
                    # Enrichir le dataframe avec les métadonnées
                    enriched_df = recs_df.copy()
                    enriched_df['image_url'] = enriched_df['asin'].map(lambda x: metadata.get(x, {}).get('image_url', ''))
                    enriched_df['rating'] = enriched_df['asin'].map(lambda x: metadata.get(x, {}).get('average_rating', 0))
                    enriched_df['store'] = enriched_df['asin'].map(lambda x: metadata.get(x, {}).get('store', ''))
                    enriched_df['price'] = enriched_df['asin'].map(lambda x: metadata.get(x, {}).get('price', ''))
                    
                    st.dataframe(
                        enriched_df[['asin', 'title', 'rating', 'store', 'price']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            'rating': st.column_config.NumberColumn('Note', format="%.1f ⭐"),
                            'price': st.column_config.TextColumn('Prix'),
                            'store': st.column_config.TextColumn('Magasin')
                        }
                    )
                
                # Option de téléchargement
                csv = recs_df.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger les recommandations (CSV)",
                    data=csv,
                    file_name=f"recommandations_{model_choice.lower().replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ Aucune recommandation n'a pu être générée.")
        
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Erreur lors de la génération des recommandations: {str(e)}")

if __name__ == "__main__":
    main()