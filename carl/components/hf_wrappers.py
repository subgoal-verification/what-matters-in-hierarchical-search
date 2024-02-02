from transformers import Trainer


class SupervisedHFWrapper(Trainer):
    def __init__(self, component_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.component_name = component_name
