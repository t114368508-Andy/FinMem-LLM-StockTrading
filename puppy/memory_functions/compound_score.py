class LinearCompoundScore:
    # 綜合分數 = 新近度 + min(重要性,100)/100;min() 是為了不讓重要性(可能超過 100)蓋過新近度(固定 0~1),兩者才公平相加
    def recency_and_importance_score(
        self, recency_score: float, importance_score: float
    ) -> float:
        importance_score = min(importance_score, 100)
        return recency_score + importance_score / 100

    # Retrieve 排名用的最終分數 = 相似度 + 綜合分數(不是只看綜合分數)
    def merge_score(
        self, similarity_score: float, recency_and_importance: float
    ) -> float:
        return similarity_score + recency_and_importance
