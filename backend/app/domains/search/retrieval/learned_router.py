import pickle
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# The specific action space mapping
ACTIONS = {
    0: {"name": "CONFIG_A", "bm25": 100, "dense": 100},
    1: {"name": "CONFIG_B", "bm25": 100, "dense": 25},
    2: {"name": "CONFIG_C", "bm25": 25, "dense": 100},
    3: {"name": "CONFIG_D", "bm25": 100, "dense": 0},
    4: {"name": "CONFIG_E", "bm25": 0, "dense": 100},
    5: {"name": "CONFIG_F", "bm25": 25, "dense": 25},
}

class RoutingModel:
    def predict(self, features: dict) -> dict:
        raise NotImplementedError

class FixedRoutingModel(RoutingModel):
    def predict(self, features: dict) -> dict:
        return {"bm25_depth": 100, "dense_depth": 100, "rrf_k": 10}

class LearnedRoutingModel(RoutingModel):
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Default to the training output location for now
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            model_path = os.path.join(base_dir, "scripts", "router_training", "router.pkl")
            
        try:
            with open(model_path, "rb") as f:
                self.model, self.feature_keys = pickle.load(f)
            self.ready = True
        except Exception as e:
            logger.error(f"Failed to load LearnedRoutingModel from {model_path}: {e}")
            self.ready = False

    def predict(self, features: dict) -> dict:
        if not self.ready:
            return FixedRoutingModel().predict(features)
            
        try:
            feature_vector = np.array([features.get(k, 0.0) for k in self.feature_keys]).reshape(1, -1)
            pred_idx = self.model.predict(feature_vector)[0]
            action = ACTIONS.get(pred_idx, ACTIONS[0])
            
            return {
                "bm25_depth": action["bm25"],
                "dense_depth": action["dense"],
                "rrf_k": 10
            }
        except Exception as e:
            logger.error(f"Learned routing prediction failed, falling back to Fixed: {e}")
            return FixedRoutingModel().predict(features)

# Global singleton
_learned_router = LearnedRoutingModel()

def get_learned_router():
    return _learned_router
