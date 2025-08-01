from dataclasses import dataclass


@dataclass
class ParallelConfig:
    dp_degree: int = -1
    sp_degree: int = 1
