# access_counter 本身是累加值(不是每次固定 +1),所以同一則記憶被引用越多次,加分速度會越來越快;
# 這個函式本身不會被單純的「引用」觸發,只有 memorydb.py 的 update_access_count_with_feed_back() 帶著實際損益回饋才會呼叫到它
class LinearImportanceScoreChange:
    def __call__(self, access_counter: int, importance_score: float) -> float:
        importance_score += access_counter * 5
        return importance_score
