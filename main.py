import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Tuple
import json
from pydantic import BaseModel
from typing import List, Optional

class DataStore:
    def __init__(self, recipes_csv: str, ratings_csv: str):
        self.recipes = pd.read_csv(recipes_csv)
        self.ratings = pd.read_csv(ratings_csv)

        self.recipes.columns = [c.strip().lower() for c in self.recipes.columns]
        self.ratings.columns = [c.strip().lower() for c in self.ratings.columns]

        assert 'id' in self.recipes.columns or 'food_id' in self.recipes.columns, "recipes.csv must have 'id' or 'food_id'"
        if 'food_id' not in self.recipes.columns:
            self.recipes.rename(columns={'id': 'food_id'}, inplace=True)

        for col in ['user_id', 'food_id', 'rating']:
            assert col in self.ratings.columns, f"ratings.csv must have column '{col}'"

        for col in ['dish_name','description','cooking_method','ingredients','dish_type']:
            if col in self.recipes.columns:
                self.recipes[col] = self.recipes[col].fillna('')

        for col in ['cooking_time', 'calories']:
            if col in self.recipes.columns:
                self.recipes[col] = pd.to_numeric(self.recipes[col], errors='coerce')

        self.recipe_ids = self.recipes['food_id'].astype(int).tolist()
        self.recipe_id_to_idx = {rid: i for i, rid in enumerate(self.recipe_ids)}

        self.user_ids = sorted(self.ratings['user_id'].unique().tolist())
        self.user_id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        
        print(f"📊 Data loaded successfully!")
        print(f"   - Recipes: {len(self.recipes)} items")
        print(f"   - Ratings: {len(self.ratings)} ratings")
        print(f"   - Users: {len(self.user_ids)} unique users")
        print(f"   - Recipe columns: {list(self.recipes.columns)}")

class ContentModel:
    def __init__(self, recipes: pd.DataFrame):
        self.recipes = recipes
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100000)
        self.tfidf = None  # sparse matrix [n_items, n_feats]

    def _compose_text(self, row: pd.Series) -> str:
        """Combine available text features into one string"""
        parts = []
        for col in ['ingredients','dish_name','description','cooking_method']:
            if col in row and isinstance(row[col], str):
                parts.append(row[col])
        return ' . '.join(parts)

    def fit(self):
        """Build TF-IDF vectors for all recipes"""
        texts = self.recipes.apply(self._compose_text, axis=1).tolist()
        self.tfidf = self.vectorizer.fit_transform(texts)
        print(f"🔧 Content model fitted:")
        print(f"   - TF-IDF shape: {self.tfidf.shape}")
        print(f"   - Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return self

    def query_from_ingredients(self, ingredients: List[str], topk: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) of topk items given a list of ingredients (or keywords)."""
        query = ' '.join(ingredients)
        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.tfidf).ravel()
        idx = np.argpartition(-sims, range(min(topk, len(sims))))[:topk]
        order = np.argsort(-sims[idx])
        return idx[order], sims[idx][order]
    
class CollabModel:
    def __init__(self, ratings: pd.DataFrame, recipe_id_to_idx: Dict[int,int], user_id_to_idx: Dict[int,int]):
        self.ratings = ratings
        self.recipe_id_to_idx = recipe_id_to_idx
        self.user_id_to_idx = user_id_to_idx
        self.R = None              # user-item CSR matrix
        self.item_cosine = None    # item-item similarity (sparse)

    def fit(self, min_interactions_per_item: int = 2):
        """Build user-item matrix and compute item-item similarities"""
        # Map ids to indices
        valid = self.ratings[self.ratings['food_id'].isin(self.recipe_id_to_idx.keys()) &
                            self.ratings['user_id'].isin(self.user_id_to_idx.keys())]
        ui = valid['user_id'].map(self.user_id_to_idx).astype(int).to_numpy()
        ii = valid['food_id'].map(self.recipe_id_to_idx).astype(int).to_numpy()
        vv = valid['rating'].astype(float).to_numpy()

        n_users = len(self.user_id_to_idx)
        n_items = len(self.recipe_id_to_idx)
        self.R = sparse.coo_matrix((vv, (ui, ii)), shape=(n_users, n_items)).tocsr()

        # Normalize by item mean to reduce user scale bias
        item_means = np.asarray(self.R.mean(axis=0)).ravel()
        R_centered = self.R - sparse.csr_matrix(np.tile(item_means, (self.R.shape[0], 1)))

        # Compute item-item cosine similarity
        print("🔧 Computing item-item similarities...")
        sims = cosine_similarity(R_centered.T, dense_output=False)
        sims.setdiag(0.0)
        self.item_cosine = sims.tocsr()
        
        print(f"✅ Collaborative model fitted:")
        print(f"   - User-item matrix shape: {self.R.shape}")
        print(f"   - Valid ratings: {len(valid)}")
        print(f"   - Sparsity: {(1 - self.R.nnz / (n_users * n_items)) * 100:.2f}%")
        return self

    def recommend_for_user(self, user_id: int, topk: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Get recommendations for a specific user"""
        if user_id not in self.user_id_to_idx:
            # cold-start user → empty
            print(f"⚠️  User {user_id} not found in training data (cold start)")
            return np.array([], dtype=int), np.array([])
            
        uidx = self.user_id_to_idx[user_id]
        user_row = self.R.getrow(uidx)  # shape [1, n_items]
        
        # Score by item-based CF: s = R_u * S_item
        scores = user_row.dot(self.item_cosine).toarray().ravel()
        
        # Remove already-rated items
        rated_items = user_row.indices
        scores[rated_items] = -np.inf
        
        top = np.argpartition(-scores, range(min(topk, np.isfinite(scores).sum())))[:topk]
        order = np.argsort(-scores[top])
        return top[order], scores[top][order]
    
class HybridRecommender:
    def __init__(self, data):
        self.data = data
        print("🔧 Building content-based model...")
        self.content = ContentModel(data.recipes).fit()
        print("🔧 Building collaborative filtering model...")
        self.collab = CollabModel(data.ratings, data.recipe_id_to_idx, data.user_id_to_idx).fit()
        print("✅ Hybrid recommender initialized!")

    def _apply_filters(self, candidates: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply requirement filters"""
        df = candidates
        if not filters:
            return df
            
        print(f"🔍 Applying filters: {filters}")
        original_count = len(df)
        
        if 'dish_type' in filters and 'dish_type' in df.columns and filters['dish_type']:
            ft = filters['dish_type']
            if isinstance(ft, (list, tuple, set)):
                df = df[df['dish_type'].isin(ft)]
            else:
                df = df[df['dish_type'] == ft]
            
        if 'max_cooking_time' in filters and 'cooking_time' in df.columns and filters['max_cooking_time'] is not None:
            df = df[df['cooking_time'] <= float(filters['max_cooking_time'])]
            
        if 'max_calories' in filters and 'calories' in df.columns and filters['max_calories'] is not None:
            df = df[df['calories'] <= float(filters['max_calories'])]
            
        print(f"   Filtered from {original_count} to {len(df)} items")
        return df

    def recommend(
        self,
        basis: str = 'hybrid',
        user_id: Optional[int] = None,
        detected_ingredients: Optional[List[str]] = None,
        k: int = 10,
        filters: Optional[Dict] = None,
        alpha: float = 0.6,
    ) -> List[Dict]:
        """
        Return a ranked list of recipes.
        basis: 'content' | 'user' | 'hybrid'
        alpha: weight for content score when basis='hybrid'
        """
        basis = basis.lower()
        print(f"\n🎯 Generating recommendations with basis='{basis}', k={k}, alpha={alpha}")
        
        recipes_df = self.data.recipes.copy()

        # --- Content scores ---
        content_idx = np.array([], dtype=int)
        content_scores = np.array([])
        if basis in ('content','hybrid'):
            print("📚 Computing content-based scores...")
            # If ingredients not provided, build a query from user's highly rated items (if any)
            if (not detected_ingredients) and (user_id is not None):
                # use top-N items rated by user as pseudo-ingredients (from titles)
                u = self.data.user_id_to_idx.get(user_id, None)
                if u is not None:
                    # get items user liked most
                    ur = self.collab.R.getrow(u)
                    liked = ur.toarray().ravel()
                    top_items = liked.argsort()[::-1][:5]
                    pseudo_terms = []
                    for idx in top_items:
                        name = recipes_df.iloc[idx]['dish_name'] if 'dish_name' in recipes_df.columns else ''
                        pseudo_terms.append(str(name))
                    detected_ingredients = pseudo_terms or detected_ingredients
                    print(f"   Using user's liked dishes as query: {pseudo_terms[:3]}...")
                    
            if not detected_ingredients:
                # final fallback: use common Vietnamese cooking terms
                detected_ingredients = ['thịt', 'rau', 'nước mắm']
                print(f"   Using default ingredients: {detected_ingredients}")
            else:
                print(f"   Using provided ingredients: {detected_ingredients}")
                
            content_idx, content_scores = self.content.query_from_ingredients(detected_ingredients, topk=max(k*5, 50))
            print(f"   Found {len(content_idx)} content candidates")

        # --- Collaborative scores ---
        collab_idx = np.array([], dtype=int)
        collab_scores = np.array([])
        if basis in ('user','hybrid') and (user_id is not None):
            print(f"👥 Computing collaborative scores for user {user_id}...")
            collab_idx, collab_scores = self.collab.recommend_for_user(user_id, topk=max(k*5, 50))
            print(f"   Found {len(collab_idx)} collaborative candidates")

        # Combine candidates
        cand_indices = set()
        for arr in [content_idx, collab_idx]:
            cand_indices.update(arr.tolist())
            
        if not cand_indices:
            # no info at all — return popular items by average rating
            print("📈 Falling back to popularity-based recommendations")
            pop = self.data.ratings.groupby('food_id')['rating'].mean().sort_values(ascending=False)
            cand_ids = [rid for rid in pop.index if rid in self.data.recipe_id_to_idx]
            cand_indices = [self.data.recipe_id_to_idx[rid] for rid in cand_ids[:max(k*5, 50)]]
        else:
            cand_indices = list(cand_indices)

        print(f"🔗 Total candidates: {len(cand_indices)}")

        # Build score vectors aligned to candidates
        idx_to_rank = {idx: i for i, idx in enumerate(cand_indices)}
        s_content = np.zeros(len(cand_indices), dtype=float)
        s_collab = np.zeros(len(cand_indices), dtype=float)
        
        for i, idx in enumerate(content_idx):
            if idx in idx_to_rank:
                s_content[idx_to_rank[idx]] = content_scores[i]
                
        for i, idx in enumerate(collab_idx):
            if idx in idx_to_rank:
                s_collab[idx_to_rank[idx]] = collab_scores[i]

        # Combine scores based on basis
        if basis == 'content':
            s_final = s_content
            print("📊 Using content-based scores only")
        elif basis == 'user':
            s_final = s_collab
            print("📊 Using collaborative scores only")
        else:
            # normalize to [0,1] before mixing
            def norm(x):
                x = x.copy()
                if np.nanmax(x) > 0:
                    x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-8)
                return x
            s_final = alpha * norm(s_content) + (1 - alpha) * norm(s_collab)
            print(f"📊 Using hybrid scores (α={alpha} content + {1-alpha} collaborative)")

        # Rank candidates
        order = np.argsort(-s_final)
        ranked_idx = [cand_indices[i] for i in order]
        ranked_df = self.data.recipes.iloc[ranked_idx].copy()
        ranked_df['__score'] = s_final[order]

        # Apply filters
        ranked_df = self._apply_filters(ranked_df, filters or {})

        # Take top-k
        top = ranked_df.head(k)
        cols = [c for c in ['food_id','dish_name','dish_type','ingredients','cooking_time','calories','description', 'image_link'] if c in top.columns]
        cols = cols + ['__score']
        out = top[cols]
        out.rename(columns={'food_id': 'id', '__score': 'score'}, inplace=True)
        
        print(f"✅ Generated {len(out)} final recommendations")
        return out.to_dict(orient='records')
    
# ---------- FastAPI ----------
class RecommendRequest(BaseModel):
    basis: str = "hybrid"  # 'content' | 'user' | 'hybrid'
    user_id: int = 0
    k: int = 10
    detected_ingredients: Optional[List[str]] = ["thịt", "rau"]
    filters: Optional[dict] = {
            "max_cooking_time": 45,
            "max_calories": 600
        }
    alpha: Optional[float] = 0.6

    # Tạo example cho Swagger UI
    class Config:
        schema_extra = {
            "example": {
                "basis": "hybrid",
                "user_id": 101,
                "k": 10,
                "detected_ingredients": ["thịt bò", "hành lá"],
                "filters": {
                    "food_type": "món nước",
                    "max_cooking_time": 45,
                    "max_calories": 700
                },
                "alpha": 0.6
            }
        }

try:
    from fastapi import FastAPI, Body
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Vietnamese Hybrid Food Recommender")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],           # Cho phép tất cả domain
        allow_credentials=True,
        allow_methods=["*"],           # Cho phép tất cả phương thức (GET, POST, ...)
        allow_headers=["*"],           # Cho phép tất cả header
    )
    _GLOBAL_MODEL: Optional[HybridRecommender] = None

    @app.on_event("startup")
    def _load_default():
        import os
        rec_path = os.getenv('RECIPES_CSV', 'foods.csv')
        rat_path = os.getenv('RATINGS_CSV', 'ratings.csv')
        if os.path.exists(rec_path) and os.path.exists(rat_path):
            data = DataStore(rec_path, rat_path)
            global _GLOBAL_MODEL
            _GLOBAL_MODEL = HybridRecommender(data)
        else:
            print("❌ RECIPES_CSV or RATINGS_CSV not found. Please set environment variables or load data manually.")
            _GLOBAL_MODEL = None

    @app.post("/recommend")
    def recommend(payload: RecommendRequest):
        assert _GLOBAL_MODEL is not None, "Model not loaded. Set RECIPES_CSV and RATINGS_CSV env vars before starting."

        basis = payload.basis.lower()
        user_id = payload.user_id
        k = payload.k
        detected_ingredients = payload.detected_ingredients
        filters = payload.filters
        alpha = payload.alpha

        results = _GLOBAL_MODEL.recommend(
            basis=basis,
            user_id=user_id,
            detected_ingredients=detected_ingredients,
            k=k,
            filters=filters,
            alpha=alpha
        )

        def clean_nan(obj):
            if isinstance(obj, list):
                return [clean_nan(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return 0
            return obj
        
        results = clean_nan(results)

        json_compatible = jsonable_encoder({
            "basis": basis,
            "results": results
        })

        return JSONResponse(content=json_compatible)
except Exception:
    app = None