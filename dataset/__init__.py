from .muti_task_dataset import MutiTaskDataset
from .rec_dataset import RecDataset
from .weighted_dataset import WeightedDataset
from .finish_dataset import FinishDataset
from .finish_adaptive_dataset import AdaptiveFinishDataset
from .duration_dataset import DurationDataset
from .multi_tower_dataset import MultiTowerDataset
from .neg_dataset import NegDataset
from .recall_dataset import RecallDataset


def build_dataset(name):
    mapper = {
        'muti_task': MutiTaskDataset,
        'weighted': WeightedDataset,
        'rec': RecDataset,
        'finish': FinishDataset,
        'adaptive_finish': AdaptiveFinishDataset,
        'negative': NegDataset,
        'duration': DurationDataset,
        'multi_tower': MultiTowerDataset,
        'recall': RecallDataset
    }
    return mapper.get(name, RecDataset)
