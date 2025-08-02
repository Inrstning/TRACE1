import torch
import numpy as np
from typing import Set, List, Tuple, Dict, Any, Optional, Union


def filter_entities_by_time(
    triples_with_time: torch.Tensor,
    batch: torch.Tensor,
    num_entities: int,
    filter_mode: str = "both",
    only_same_timestamp: bool = True
) -> Dict[Tuple[int, int, int], Set[int]]:
    """
    Build time-aware filtering index structure.
    
    Args:
        triples_with_time: Quadruples with timestamps [s, p, o, t]
        batch: Current evaluation batch
        num_entities: Total number of entities
        filter_mode: Filtering mode, "head"(only filter head entities), "tail"(only filter tail entities) or "both"(filter both)
        only_same_timestamp: Whether to only filter triples with the same timestamp
    
    Returns:
        A dictionary with keys (s,p,t) and values as sets of entities
    """
    # Create dictionary for filtering
    to_filter = {}
    
    # Convert to numpy array on CPU for easier processing
    if isinstance(triples_with_time, torch.Tensor):
        triples_with_time = triples_with_time.cpu().numpy()
    
    # Build time-aware filtering index
    for quad in triples_with_time:
        s, p, o, t = map(int, quad)
        
        # Filter tail entities
        if filter_mode in ["tail", "both"]:
            if (s, p, t) not in to_filter:
                to_filter[(s, p, t)] = set()
            to_filter[(s, p, t)].add(o)
        
        # Filter head entities
        if filter_mode in ["head", "both"]:
            if (o, p, t) not in to_filter:
                to_filter[(o, p, t)] = set()
            to_filter[(o, p, t)].add(s)
    
    return to_filter


def get_time_aware_mask(
    batch: torch.Tensor,
    to_filter: Dict[Tuple[int, int, int], Set[int]],
    num_entities: int,
    entity_pos: str = "tail",
) -> torch.Tensor:
    """
    Get time-aware filtering mask matrix
    
    Args:
        batch: Current evaluation batch, shape [batch_size, 4], representing [s, p, o, t]
        to_filter: Time-aware filtering index
        num_entities: Total number of entities
        entity_pos: Position of entity to predict, "head" or "tail"
        
    Returns:
        Filtering mask, shape [batch_size, num_entities], 1 indicates positions to filter
    """
    batch_size = batch.shape[0]
    device = batch.device
    
    # Create initial mask matrix, all zeros
    mask = torch.zeros(batch_size, num_entities, dtype=torch.bool, device=device)
    
    # Apply filtering for each sample
    for i in range(batch_size):
        if entity_pos == "tail":
            s, p, o, t = batch[i]
            s, p, t = s.item(), p.item(), t.item()
            
            # Get all tail entities for current (s,p,t)
            if (s, p, t) in to_filter:
                for obj in to_filter[(s, p, t)]:
                    if obj != o.item():  # Don't filter current evaluation object
                        mask[i, obj] = True
        else:  # entity_pos == "head"
            s, p, o, t = batch[i]
            o, p, t = o.item(), p.item(), t.item()
            
            # Get all head entities for current (o,p,t)
            if (o, p, t) in to_filter:
                for subj in to_filter[(o, p, t)]:
                    if subj != s.item():  # Don't filter current evaluation object
                        mask[i, subj] = True
    
    return mask


def filter_scores_by_time(
    scores: torch.Tensor,
    batch: torch.Tensor,
    to_filter: Dict[Tuple[int, int, int], Set[int]],
    entity_pos: str = "tail",
    filter_value: float = float("-Inf"),
) -> torch.Tensor:
    """
    Filter score matrix based on time-aware filtering index
    
    Args:
        scores: Prediction score matrix, shape [batch_size, num_entities]
        batch: Current evaluation batch, shape [batch_size, 4], representing [s, p, o, t]
        to_filter: Time-aware filtering index
        entity_pos: Position of entity to predict, "head" or "tail"
        filter_value: Value to replace filtered positions, usually negative infinity
        
    Returns:
        Filtered score matrix
    """
    # Copy score matrix to avoid modifying original data
    filtered_scores = scores.clone()
    batch_size = batch.shape[0]
    
    # Get filtering mask
    mask = get_time_aware_mask(batch, to_filter, scores.shape[1], entity_pos)
    
    # Apply mask for filtering
    filtered_scores[mask] = filter_value
    
    return filtered_scores


def compute_time_aware_filtered_ranks(
    scores: torch.Tensor,
    batch: torch.Tensor,
    to_filter: Dict[Tuple[int, int, int], Set[int]],
    entity_pos: str = "tail",
) -> torch.Tensor:
    """
    Compute time-aware filtered ranks (modified to filter entities first then sort)

    Args:
        scores: Prediction score matrix, shape [batch_size, num_entities]
        batch: Current evaluation batch, shape [batch_size, 4], representing [s, p, o, t]
        to_filter: Time-aware filtering index, key=(s,p,t) or (o,p,t), value=set of entities to filter
        entity_pos: Position of entity to predict, "head" or "tail"

    Returns:
        Filtered rank tensor, shape [batch_size]
    """
    batch_size = batch.shape[0]
    num_entities = scores.shape[1]
    device = scores.device
    ranks = torch.zeros(batch_size, dtype=torch.long, device=device)

    targets = batch[:, 0] if entity_pos == "head" else batch[:, 2]

    for i in range(batch_size):
        target_entity = targets[i].item()
        s, p, o, t = map(int, batch[i].tolist()) # Ensure integers

        # Determine filter key and set of entities to filter
        filter_key = (s, p, t) if entity_pos == "tail" else (o, p, t)
        entities_to_filter_set = to_filter.get(filter_key, set())

        # Build valid candidate entity list (all entities except filtered ones, but keep target entity)
        valid_candidate_indices = []
        for entity_idx in range(num_entities):
            if entity_idx == target_entity or entity_idx not in entities_to_filter_set:
                valid_candidate_indices.append(entity_idx)

        # Convert to Tensor
        valid_candidate_indices_tensor = torch.tensor(valid_candidate_indices, device=device, dtype=torch.long)

        # Get scores for valid candidate entities
        scores_for_valid_candidates = scores[i][valid_candidate_indices_tensor]

        # Find target entity index in valid candidate list (relative to valid_candidate_indices_tensor)
        # Use nonzero(as_tuple=True)[0] to ensure we get an index tensor
        target_entity_rank_in_candidates = (valid_candidate_indices_tensor == target_entity).nonzero(as_tuple=True)[0]

        # Sort valid candidate entity scores in descending order
        _, sorted_indices_within_candidates = torch.sort(scores_for_valid_candidates, descending=True)

        # Find target entity rank in sorted list (1-based)
        # Use nonzero(as_tuple=True)[0].item() to get rank value
        rank = (sorted_indices_within_candidates == target_entity_rank_in_candidates).nonzero(as_tuple=True)[0].item() + 1
        ranks[i] = rank

    return ranks


def compute_time_aware_metrics(ranks: torch.Tensor, hits_at_k: List[int] = [1, 3, 10]) -> Dict[str, float]:
    """
    Compute time-aware filtering metrics
    
    Args:
        ranks: Rank tensor
        hits_at_k: List of K values to compute Hits@K
        
    Returns:
        Dictionary containing MRR and Hits@K metrics
    """
    metrics = {}
    
    # Compute MRR (Mean Reciprocal Rank)
    metrics["time_aware_mean_reciprocal_rank"] = (1.0 / ranks.float()).mean().item()
    
    # Compute various Hits@K
    for k in hits_at_k:
        metrics[f"time_aware_hits_at_{k}"] = (ranks <= k).float().mean().item()
    
    return metrics 