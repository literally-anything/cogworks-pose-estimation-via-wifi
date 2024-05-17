from torch.optim import RMSprop

from verbose_spoon_fearless_butter.model import net

optimizer = RMSprop(net.parameters(), lr=1e-4)