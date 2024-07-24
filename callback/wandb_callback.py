from transformers.integrations import WandbCallback


class CustomWandbCallback(WandbCallback):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    # def setup(self, args, state, model, **kwargs):
    #     super().setup(args, state, model, **kwargs)
    #     # Log all kwargs
    #     self._wandb.log(self.kwargs)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        super().on_train_begin(args, state, control, model, **kwargs)
        # Log all kwargs
        self._wandb.log(self.kwargs)
