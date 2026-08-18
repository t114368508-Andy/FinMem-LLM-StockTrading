import numpy as np
from typing import Tuple


# 衰減公式:recency_factor、importance_factor 每層各自從 config.toml 讀入不同數值,不是固定常數(這裡的預設值只在沒指定時才會用到)
class ExponentialDecay:
    def __init__(
        self,
        recency_factor: float = 10.0,
        importance_factor: float = 0.988,
    ):
        self.recency_factor = recency_factor
        self.importance_factor = importance_factor

    def __call__(
        self, important_score: float, delta: float
    ) -> Tuple[float, float, float]:
        delta += 1
        new_recency_score = np.exp(-(delta / self.recency_factor))  # 新近度:指數衰減,衰減係數越小掉得越快
        new_important_score = important_score * self.importance_factor  # 重要性:每天乘上一個小於 1 的係數,只會往下掉

        return new_recency_score, new_important_score, delta
