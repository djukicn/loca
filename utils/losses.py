from torch import nn


class ObjectNormalizedL2Loss(nn.Module):

    def __init__(self):
        super(ObjectNormalizedL2Loss, self).__init__()

    def forward(self, output, dmap, num_objects):
        return ((output - dmap) ** 2).sum() / num_objects
