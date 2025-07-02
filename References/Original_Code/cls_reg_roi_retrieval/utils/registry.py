_REG = {}
def register(name):
    def _deco(fn): _REG[name] = fn; return fn
    return _deco
def get(name): return _REG[name]
