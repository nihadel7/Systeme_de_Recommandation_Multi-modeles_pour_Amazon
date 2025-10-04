import pandas as pd
import numpy as np
from recommendation_filters import content_based_filter, popularity_filter
from models import collaborative_model_based
from sklearn.metrics.pairwise import cosine_similarity

# --- Data Loading and Preparation ---
def load_data():
    print("Loading name data...")
    name_df = pd.read_json('data/asin_title.json.gz')
    
    print("Loading final processed data...")
    final_df = pd.read_json('data/traitees/final.json.gz')
    
    print("Preparing content-based data...")
    df = content_based_filter.cbf_data('data/traitees/final.json.gz')
    idx = content_based_filter.indices(df)
    cosim = content_based_filter.cosine_sim(df['description'])
    
    print("Training collaborative model...")
    svd_model = collaborative_model_based.train(
        df_path='data/traitees/final.json.gz',
        sample_frac=1.0,
        idx='asin',
        col='reviewerID', 
        val='positive_prob'
    )
    
    return name_df, final_df, df, idx, cosim, svd_model

# --- Evaluation Metrics ---
def diversity_score(recommended_list, asin_to_idx, cosine_sim):
    if len(recommended_list) < 2:
        return 0.0
    
    idxs = [asin_to_idx.get(asin) for asin in recommended_list if asin in asin_to_idx]
    idxs = [i for i in idxs if i is not None]
    
    if len(idxs) < 2:
        return 0.0
    
    sim_sum = 0
    count = 0
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            sim_sum += cosine_sim[idxs[i], idxs[j]]
            count += 1
    
    return 1 - (sim_sum / count) if count > 0 else 0.0

def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / k if k > 0 else 0

def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / len(relevant_set) if len(relevant_set) > 0 else 0

# --- Recommendation Functions ---
def recommend_popularity(df_path, rev_count=25, rating=3, sentiment=0.6):
    return popularity_filter.recommend(df_path, rev_count, rating, sentiment)

def recommend_content_based(prod_asin, cosine_sim, indices, cbf_df, lim=5, min_rate=2):
    return content_based_filter.recommend(prod_asin, cosine_sim, indices, cbf_df, lim, min_rate)

def recommend_collaborative(product, model, corr_thresh=0.3, top_n=5):
    try:
        return collaborative_model_based.recommend(product, model, corr_thresh, top_n)
    except Exception as e:
        print(f"Error in collaborative recommendation for {product}: {str(e)}")
        return []

# --- Evaluation Method ---
def evaluate_method(method_name, recommend_func, test_set, k=5, rec_kwargs=None, metric_kwargs=None):
    try:
        # Get recommendations
        recs = recommend_func(**rec_kwargs) if rec_kwargs else recommend_func()
        
        if not recs:
            return {'method': method_name, 'precision': 0, 'recall': 0, 'diversity': 0}
        
        # Calculate metrics
        prec = precision_at_k(recs, test_set, k)
        rec = recall_at_k(recs, test_set, k)
        div = diversity_score(recs, **metric_kwargs) if metric_kwargs else 0
        
        return {'method': method_name, 'precision': prec, 'recall': rec, 'diversity': div}
    except Exception as e:
        print(f"Error evaluating {method_name}: {str(e)}")
        return {'method': method_name, 'precision': 0, 'recall': 0, 'diversity': 0}

# --- Test Set Generation ---
def get_realistic_test_set(product_asin, df, cosine_sim, indices, top_k=5):
    if product_asin not in indices:
        return []
    
    idx = indices[product_asin]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_sim_idx = [i[0] for i in sim_scores[1:top_k + 1]]
    test_set = df.iloc[top_sim_idx]['asin'].tolist()
    return test_set

# --- Main Evaluation Function ---
def main_evaluation():
    print("Loading data...")
    name_df, final_df, df, idx, cosim, svd_model = load_data()
    asin_to_idx = {asin: i for i, asin in enumerate(df['asin'])}
    
    n_tests = 1000
    results = []
    products_test = name_df['asin'].sample(n=n_tests, random_state=42).tolist()
    
    # Common metric arguments
    metric_args = {
        'asin_to_idx': asin_to_idx,
        'cosine_sim': cosim
    }
    
    for i, product_asin in enumerate(products_test):
        if i % 100 == 0:
            print(f"Processing product {i}/{n_tests}")
        
        test_set = get_realistic_test_set(product_asin, df, cosim, idx)
        if not test_set:
            continue
        
        # Popularity-based evaluation
        results.append(evaluate_method(
            method_name="Popularité",
            recommend_func=recommend_popularity,
            test_set=test_set,
            k=5,
            rec_kwargs={
                'df_path': 'data/traitees/final.json.gz',
                'rev_count': 25,
                'rating': 3,
                'sentiment': 0.6
            },
            metric_kwargs=metric_args
        ))
        
        # Content-based evaluation
        results.append(evaluate_method(
            method_name="Basé contenu",
            recommend_func=recommend_content_based,
            test_set=test_set,
            k=5,
            rec_kwargs={
                'prod_asin': product_asin,
                'cosine_sim': cosim,
                'indices': idx,
                'cbf_df': df,
                'lim': 5,
                'min_rate': 2
            },
            metric_kwargs=metric_args
        ))
        
        # Collaborative evaluation
        results.append(evaluate_method(
            method_name="Collaboratif",
            recommend_func=recommend_collaborative,
            test_set=test_set,
            k=5,
            rec_kwargs={
                'product': product_asin,
                'model': svd_model,
                'corr_thresh': 0.3,
                'top_n': 10
            },
            metric_kwargs=metric_args
        ))
    
    # Save and display results
    df_results = pd.DataFrame(results)
    print("\nPerformance summary:")
    summary = df_results.groupby('method').mean(numeric_only=True)
    print(summary)
    
    df_results.to_csv('evaluation_report2.csv', index=False)
    print("Report saved to 'evaluation_report.csv'")

if __name__ == "__main__":
    main_evaluation()