"""
Dataset Recommendation Engine
Rule-based, interpretable recommendation system
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

# Simple stopwords for keyword matching
STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
             'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'how', 'do'}


def load_catalog(path: str) -> pd.DataFrame:
    """
    Load and standardize the dataset catalog.
    
    Args:
        path: Path to catalog_datasets.csv
    
    Returns:
        DataFrame with standardized columns
    """
    df = pd.read_csv(path)
    
    # Standardize keywords column
    if 'keywords' in df.columns:
        df['keywords_list'] = df['keywords'].apply(lambda x: 
            [kw.strip().lower() for kw in str(x).split(',') if kw.strip()] 
            if pd.notna(x) else []
        )
    else:
        df['keywords_list'] = [[] for _ in range(len(df))]
    
    # Standardize text columns
    text_cols = ['dataset_name', 'domain', 'provider_or_platform', 'description']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip()
    
    return df


def tokenize_query(query: str) -> List[str]:
    """
    Simple tokenization: lowercase, split, remove stopwords.
    
    Args:
        query: User's research question
    
    Returns:
        List of tokens
    """
    tokens = query.lower().split()
    tokens = [t.strip('.,!?;:') for t in tokens]
    tokens = [t for t in tokens if t and t not in STOPWORDS and len(t) > 2]
    return tokens


def score_by_keywords(query: str, row: pd.Series) -> Tuple[float, List[str]]:
    """
    Score a dataset row based on keyword matching.
    
    Args:
        query: User's research question
        row: DataFrame row with dataset info
    
    Returns:
        (score, matched_keywords)
    """
    query_tokens = tokenize_query(query)
    
    score = 0.0
    matched = []
    
    # Check keywords field (highest weight)
    keywords = row.get('keywords_list', [])
    for kw in keywords:
        for token in query_tokens:
            if token in kw or kw in token:
                score += 3.0
                if kw not in matched:
                    matched.append(kw)
                break
    
    # Check dataset name (medium weight)
    dataset_name = str(row.get('dataset_name', '')).lower()
    for token in query_tokens:
        if token in dataset_name:
            score += 2.0
            if token not in matched:
                matched.append(token)
    
    # Check domain (medium weight)
    domain = str(row.get('domain', '')).lower()
    for token in query_tokens:
        if token in domain:
            score += 1.5
            if f"domain:{token}" not in matched:
                matched.append(f"domain:{token}")
    
    # Check description (low weight)
    description = str(row.get('description', '')).lower()
    for token in query_tokens:
        if token in description:
            score += 0.5
    
    # Boost for public/easy access
    if row.get('access_level', '').lower() in ['public', 'free']:
        score *= 1.1
    
    return score, matched


def recommend_datasets(
    query: str, 
    catalog_df: pd.DataFrame, 
    top_n: int = 5,
    domain_filter: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Recommend datasets based on research question.
    
    Args:
        query: Research question
        catalog_df: Dataset catalog
        top_n: Number of recommendations
        domain_filter: Optional list of domains to filter
    
    Returns:
        DataFrame with top N recommendations, sorted by score
    """
    # Apply domain filter if specified
    df = catalog_df.copy()
    if domain_filter:
        df = df[df['domain'].isin(domain_filter)]
    
    if df.empty:
        return pd.DataFrame()
    
    # Score each dataset
    scores = []
    matched_kws = []
    
    for idx, row in df.iterrows():
        score, matched = score_by_keywords(query, row)
        scores.append(score)
        matched_kws.append(matched)
    
    df['score'] = scores
    df['matched_keywords'] = matched_kws
    
    # Filter out zero scores and sort
    df = df[df['score'] > 0].sort_values('score', ascending=False)
    
    # Return top N with relevant columns
    cols = [
        'dataset_id', 'dataset_name', 'provider_or_platform', 
        'domain', 'granularity', 'coverage', 
        'access_level', 'access_note', 
        'score', 'matched_keywords'
    ]
    
    # Only keep columns that exist
    cols = [c for c in cols if c in df.columns]
    
    return df[cols].head(top_n).reset_index(drop=True)


def get_recommendation_summary(recommendations: pd.DataFrame) -> str:
    """
    Generate a text summary of recommendations.
    
    Args:
        recommendations: DataFrame from recommend_datasets()
    
    Returns:
        Formatted summary string
    """
    if recommendations.empty:
        return "No matching datasets found."
    
    summary = f"Found {len(recommendations)} relevant dataset(s):\n\n"
    
    for idx, row in recommendations.iterrows():
        summary += f"{idx+1}. {row['dataset_name']}\n"
        summary += f"   Provider: {row['provider_or_platform']}\n"
        summary += f"   Score: {row['score']:.1f}\n"
        if row.get('matched_keywords'):
            summary += f"   Matched: {', '.join(row['matched_keywords'][:3])}\n"
        summary += "\n"
    
    return summary