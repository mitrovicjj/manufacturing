import pandas as pd
import json
import hashlib
from hashlib import md5
import pathlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class FeatureStore:
    """Cache za feature-e baziran na config parametrima"""
    
    def __init__(self, cache_dir: str = "data/feature_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, config: Dict[str, Any]) -> str:  # ← SAMO config!
        """Hash config parametara za jedinstveni cache key"""
        config_str = "_".join([f"{k}:{v}" for k, v in sorted(config.items())])
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_features(self, config: Dict[str, Any]) -> pd.DataFrame | None:  # ← SAMO config!
        """Pokušaj učitati feature-e iz cache-a"""
        cache_key = self._get_cache_key(config)  # ← BEZ data_path
        cache_path = self.cache_dir / f"features_{cache_key}.csv"
        
        if cache_path.exists():
            print(f"✅ CACHE HIT: {cache_path}")
            return pd.read_csv(cache_path)
        else:
            print(f"❌ CACHE MISS: {cache_path} - računam feature-e")
            return None
    
    def save_features(self, df: pd.DataFrame, config: Dict[str, Any]):  # ← SAMO config!
        """Snimi feature-e u cache"""
        cache_key = self._get_cache_key(config)  # ← BEZ data_path
        cache_path = self.cache_dir / f"features_{cache_key}.csv"
        df.to_csv(cache_path, index=False)
        print(f"💾 CACHE SAVED: {cache_path}")