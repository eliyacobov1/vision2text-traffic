class _Hub:
    def load(self, *args, **kwargs):
        raise NotImplementedError("PyTorch is not installed.")

hub = _Hub()
