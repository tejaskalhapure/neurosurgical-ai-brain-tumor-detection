"""Patient-Level Dataset Splitting.

Guarantees strict case-level partition to prevent patient data leakage.
Slices from the same patient volume are NEVER partitioned across splits.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
import random
import json
from dataclasses import dataclass, field
from ai.config.dataset_config import DatasetConfig


@dataclass
class DatasetSplit:
    """Case ID partitions for train, validation, and test splits."""
    train_cases: List[str]
    val_cases: List[str]
    test_cases: List[str]
    seed: int
    ratios: Tuple[float, float, float]
    total_cases: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "seed": self.seed,
            "ratios": {
                "train": self.ratios[0],
                "val": self.ratios[1],
                "test": self.ratios[2]
            },
            "counts": {
                "train": len(self.train_cases),
                "val": len(self.val_cases),
                "test": len(self.test_cases)
            },
            "train_cases": self.train_cases,
            "val_cases": self.val_cases,
            "test_cases": self.test_cases
        }


class PatientLevelSplitter:
    """Splits patient cases reproducibly into train, validation, and test sets."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.seed = self.config.seed
        self.train_ratio = self.config.train_ratio
        self.val_ratio = self.config.val_ratio
        self.test_ratio = self.config.test_ratio

    def split_cases(
        self,
        case_ids: List[str],
        seed: Optional[int] = None,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None
    ) -> DatasetSplit:
        """Partitions unique case IDs into train, validation, and test sets.

        Guarantees zero overlap between sets (prevents patient data leakage).
        """
        # Ensure unique case IDs and deterministic initial sort
        unique_cases = sorted(list(set(case_ids)))
        n_total = len(unique_cases)

        if n_total == 0:
            return DatasetSplit(
                train_cases=[], val_cases=[], test_cases=[],
                seed=seed or self.seed,
                ratios=(train_ratio or self.train_ratio, val_ratio or self.val_ratio, test_ratio or self.test_ratio),
                total_cases=0
            )

        active_seed = seed if seed is not None else self.seed
        t_rat = train_ratio if train_ratio is not None else self.train_ratio
        v_rat = val_ratio if val_ratio is not None else self.val_ratio
        te_rat = test_ratio if test_ratio is not None else self.test_ratio

        # Validate ratios
        tot = t_rat + v_rat + te_rat
        if not (0.99 <= tot <= 1.01):
            raise ValueError(f"Ratios must sum to 1.0, got {tot:.3f}")

        # Reproducible shuffle using dedicated Random instance
        rng = random.Random(active_seed)
        shuffled = list(unique_cases)
        rng.shuffle(shuffled)

        if n_total == 1:
            # Single case goes to train
            return DatasetSplit(
                train_cases=shuffled, val_cases=[], test_cases=[],
                seed=active_seed, ratios=(t_rat, v_rat, te_rat), total_cases=1
            )

        if n_total == 2:
            return DatasetSplit(
                train_cases=[shuffled[0]], val_cases=[shuffled[1]], test_cases=[],
                seed=active_seed, ratios=(t_rat, v_rat, te_rat), total_cases=2
            )

        # Calculate cutoffs
        n_train = int(round(n_total * t_rat))
        n_val = int(round(n_total * v_rat))

        # Enforce minimums when cases >= 3 and ratios > 0
        if v_rat > 0 and n_val == 0 and n_total >= 2:
            n_val = 1
            if n_train > 1:
                n_train -= 1

        n_train = max(1, min(n_train, n_total - (1 if (v_rat > 0 or te_rat > 0) else 0)))
        rem = n_total - n_train

        if te_rat > 0:
            n_val = min(n_val, rem - 1) if rem > 1 else (rem if v_rat >= te_rat else 0)
        else:
            n_val = rem

        train_set = shuffled[:n_train]
        val_set = shuffled[n_train:n_train + n_val]
        test_set = shuffled[n_train + n_val:]

        # Rigorous verification against data leakage
        assert set(train_set).isdisjoint(set(val_set)), "Data leakage: train and val overlap!"
        assert set(train_set).isdisjoint(set(test_set)), "Data leakage: train and test overlap!"
        assert set(val_set).isdisjoint(set(test_set)), "Data leakage: val and test overlap!"
        assert len(train_set) + len(val_set) + len(test_set) == n_total, "Case count mismatch!"

        return DatasetSplit(
            train_cases=sorted(train_set),
            val_cases=sorted(val_set),
            test_cases=sorted(test_set),
            seed=active_seed,
            ratios=(t_rat, v_rat, te_rat),
            total_cases=n_total
        )

    def save_split_manifest(self, split: DatasetSplit, output_path: Path) -> Path:
        """Serializes split manifests to a JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(split.to_dict(), f, indent=2)
        return out

    def load_split_manifest(self, manifest_path: Path) -> DatasetSplit:
        """Loads a split manifest from a JSON file."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ratios = (
            data["ratios"]["train"],
            data["ratios"]["val"],
            data["ratios"]["test"]
        )
        return DatasetSplit(
            train_cases=data["train_cases"],
            val_cases=data["val_cases"],
            test_cases=data["test_cases"],
            seed=data["seed"],
            ratios=ratios,
            total_cases=data["total_cases"]
        )
