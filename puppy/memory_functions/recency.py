# 新記憶誕生時,新近度一律先給滿分,不分層,之後才依各層自己的衰減係數往下掉
class R_ConstantInitialization:
    def __call__(self) -> float:
        return 1.0
