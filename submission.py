"""Select one candidate per question.

A checkpoint whose config method is "blend_mlp" mixes the listwise MLP with
the ranker-plus-linear blend. Method "blend" mixes the ranker with the
pairwise linear score. A checkpoint that contains model.txt and is neither
blend is the ranker alone. Any other checkpoint is the linear embedding score.
"""
import json
from pathlib import Path

from linear_model import choose, load_bundle


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('This method needs a checkpoint directory')
    path = Path(checkpoint)
    config_path = path / 'config.json'
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding='utf-8'))
        if config.get('method') == 'blend_close':
            from close_model import predict_close_blend
            return predict_close_blend(bank, arrays, path)
        if config.get('method') == 'blend_mlp':
            from listwise_mlp import predict_blend_mlp
            return predict_blend_mlp(bank, arrays, path)
        if config.get('method') == 'blend':
            from rank_features import predict_blend
            return predict_blend(bank, arrays, path)
    if (path / 'model.txt').is_file():
        from rank_features import predict_rank
        return predict_rank(bank, arrays, path)
    return choose(bank, arrays['embeddings'], load_bundle(path))
