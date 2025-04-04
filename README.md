# comfyui-lopi999-nodes
A collection of some nodes I got Deepseek and ChatGPT to make. Mildly useful for some randomness and a scheduler.

# Node List

## Random SDXL Latent Size
![Random SDXL Latent Size](https://i.imgur.com/bJaDdtu.png)

This node is sort of the one that started this collection of nodes. It is a fairly straightforward node based heavily on the Empty Latent Picker node from [ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials). Most of the functions are pretty self-explanitory. Landscape and portrait resolutions can either be included or excluded, and one can define what the minimum landscape and the maximum landscape resolutions should be.

## Advanced Text Switch
![Advanced Text Switch](https://i.imgur.com/P7x7WHT.png)

Take the text switch node, and use built-in multiline textboxes instead of text inputs, allow for easy randomization and concatenation, and you get the Advanced Text Switch node. Technically should have code to hide textboxes that aren't in use but I guess I wasn't able to get the whale to properly code that in. Otherwise, it is perfectly functional, helps trim down on node counts when using multiple text inputs, and allows for easy randomization. Perfect for switching around artist mixes at random if using Illu/NoobAI models.

## Random Normal Distribution
![Random Normal Distribution](https://i.imgur.com/S3h4v4A.png)

This node may be useful to someone. All random number generator nodes for ComfyUI seem to only follow a uniform distribution, this instead gives you one with a normal distribution, with the ability to define a min/max and the mean/std.

## Zipf Scheduler
![Zipf Scheduler](https://i.imgur.com/prueISd.png)

Provides a unique scheduler that I fully titled as zipf_linear. It follows [Zipf's Law](https://en.wikipedia.org/wiki/Zipf's_law) as a method to remove noise. It is somewhat similar to the exponential scheduler, though a bit different as it has a heavier tail. Needs around 25-30+ steps for good results.
