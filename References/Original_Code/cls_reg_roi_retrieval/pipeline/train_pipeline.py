from cls_reg_roi_retrieval.core import train as core_train
def run(cfg=None, epochs=1):
    core_train.main(cfg, extra_epochs=epochs)
