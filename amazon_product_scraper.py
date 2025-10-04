import requests
from bs4 import BeautifulSoup
import json
import time
from typing import List, Dict

# Configuration des headers pour simuler un navigateur
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7'
}

# Délai entre les requêtes pour éviter le blocage
REQUEST_DELAY = 2

def scrape_amazon_products(product_urls: List[str]) -> List[Dict]:
    """Scrape les données de plusieurs produits Amazon"""
    products_data = []
    
    for url in product_urls:
        print(f"Traitement de l'URL: {url}")
        product_data = scrape_amazon_product(url)
        
        if product_data:
            products_data.append(product_data)
            print(f"Données extraites pour: {product_data.get('title', 'Produit inconnu')}")
        else:
            print(f"Échec de l'extraction pour l'URL: {url}")
        
        # Respecter un délai entre les requêtes
        time.sleep(REQUEST_DELAY)
    
    return products_data

def scrape_amazon_product(url: str) -> Dict:
    """Scrape les données d'un seul produit Amazon"""
    try:
        # Faire la requête HTTP
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        # Parser le HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extraire les données
        product_data = {
            'url': url,
            'title': get_product_title(soup),
            'price': get_product_price(soup),
            'original_price': get_original_price(soup),
            'rating': get_product_rating(soup),
            'review_count': get_review_count(soup),
            'availability': get_availability(soup),
            'description': get_description(soup),
            'features': get_product_features(soup),
            'image_url': get_product_image(soup),
            'brand': get_product_brand(soup),
            'seller': get_product_seller(soup),
            'asin': get_product_asin(url),
            'categories': get_product_categories(soup)
        }
        
        return product_data
        
    except Exception as e:
        print(f"Une erreur s'est produite pour {url}: {str(e)}")
        return None

# Fonctions d'extraction (certaines sont nouvelles)
def get_product_title(soup):
    try:
        title = soup.find('span', {'id': 'productTitle'}).get_text().strip()
        return title
    except:
        return None

def get_product_price(soup):
    try:
        # Essayer différents sélecteurs
        price_whole = soup.find('span', {'class': 'a-price-whole'}).get_text().strip()
        price_fraction = soup.find('span', {'class': 'a-price-fraction'}).get_text().strip()
        return f"{price_whole}{price_fraction} €"
    except:
        try:
            price = soup.find('span', {'id': 'priceblock_ourprice'}).get_text().strip()
            return price
        except:
            return None

def get_original_price(soup):
    try:
        price = soup.find('span', {'class': 'a-price a-text-price'}).find('span', {'class': 'a-offscreen'}).get_text().strip()
        return price
    except:
        return None

def get_product_rating(soup):
    try:
        rating = soup.find('span', {'class': 'a-icon-alt'}).get_text().strip().split()[0]
        return float(rating.replace(',', '.'))
    except:
        return None

def get_review_count(soup):
    try:
        count = soup.find('span', {'id': 'acrCustomerReviewText'}).get_text().strip().split()[0]
        return int(count.replace('.', '').replace(',', ''))
    except:
        return None

def get_availability(soup):
    try:
        availability = soup.find('div', {'id': 'availability'}).get_text().strip()
        return availability
    except:
        return None

def get_description(soup):
    try:
        description = soup.find('div', {'id': 'productDescription'}).get_text().strip()
        return description
    except:
        return None

def get_product_features(soup):
    try:
        features = []
        feature_list = soup.find('div', {'id': 'feature-bullets'}).find_all('li')
        for feature in feature_list:
            features.append(feature.get_text().strip())
        return features
    except:
        return None

def get_product_image(soup):
    try:
        image = soup.find('img', {'id': 'landingImage'})['src']
        return image
    except:
        return None

def get_product_brand(soup):
    try:
        brand = soup.find('a', {'id': 'bylineInfo'}).get_text().strip()
        return brand
    except:
        try:
            brand = soup.find('a', {'id': 'brand'}).get_text().strip()
            return brand
        except:
            return None

def get_product_seller(soup):
    try:
        seller = soup.find('div', {'id': 'merchant-info'}).get_text().strip()
        return seller
    except:
        return None

def get_product_asin(url):
    try:
        # Extraire l'ASIN de l'URL
        asin = url.split('/dp/')[1].split('/')[0]
        return asin
    except:
        return None

def get_product_categories(soup):
    try:
        categories = []
        breadcrumbs = soup.find('div', {'id': 'wayfinding-breadcrumbs_container'}).find_all('li')
        for item in breadcrumbs:
            categories.append(item.get_text().strip())
        return categories
    except:
        return None

# Exemple d'utilisation avec plusieurs produits
if __name__ == "__main__":
    # Liste d'URLs de produits Amazon (exemples)
    product_urls = [
        "https://www.amazon.fr/dp/B08N5KWB9H",  # Exemple produit 1
        "https://www.amazon.fr/dp/B07PFFMP9P",  # Exemple produit 2
        "https://www.amazon.fr/dp/B08H95Y452",  # Exemple produit 3
        "https://www.amazon.fr/dp/B07DJD1RTM"   # Exemple produit 4
    ]
    
    print(f"Début de l'extraction pour {len(product_urls)} produits...")
    all_products_data = scrape_amazon_products(product_urls)
    
    if all_products_data:
        print("\nExtraction terminée avec succès!")
        print(f"Nombre de produits extraits: {len(all_products_data)}")
        
        # Sauvegarder les résultats dans un fichier JSON
        with open('amazon_products_data.json', 'w', encoding='utf-8') as f:
            json.dump(all_products_data, f, indent=2, ensure_ascii=False)
        
        print("Données sauvegardées dans 'amazon_products_data.json'")
        
        # Afficher un résumé
        print("\nRésumé des produits extraits:")
        for i, product in enumerate(all_products_data, 1):
            print(f"\nProduit {i}:")
            print(f"Titre: {product.get('title')}")
            print(f"Prix: {product.get('price')}")
            print(f"Note: {product.get('rating')}/5 ({product.get('review_count')} avis)")
            print(f"Marque: {product.get('brand')}")
    else:
        print("Aucune donnée n'a pu être extraite.")