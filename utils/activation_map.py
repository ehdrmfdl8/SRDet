def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    return img

class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.model = model
        self.handlers = []

class feature_maps(_BaseWrapper):
    def __init__(self, model, candidate_layers=None):
        super(feature_maps, self).__init__(model)
        self.fmap_pool = {}
        self.candidate_layers = candidate_layers

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output[0].detach()
            return forward_hook

        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def get_fmaps(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer) # feature map
        return fmaps