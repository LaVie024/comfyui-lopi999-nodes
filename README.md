# comfyui-lopi999-nodes
Utility nodes for some randomness in your workflows, like random latent sizes. Also includes two schedulers that can be used with any KSampler node.

# Node List

## Random SDXL Latent Size
![Random SDXL Latent Size](https://i.imgur.com/n1xiaKh.png)

This node is sort of the one that started this collection of nodes. It is a fairly straightforward node based heavily on the Empty Latent Picker node from [ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials). Most of the functions are pretty self-explanitory. Landscape and portrait resolutions can either be included or excluded, and one can define what the minimum landscape and the maximum landscape resolutions should be.

## Advanced Text Switch
![Advanced Text Switch](https://i.imgur.com/P7x7WHT.png)

Take the text switch node, and use built-in multiline textboxes instead of text inputs, allow for easy randomization and concatenation, and you get the Advanced Text Switch node. Technically should have code to hide textboxes that aren't in use but I guess I wasn't able to get the whale to properly code that in. Otherwise, it is perfectly functional, helps trim down on node counts when using multiple text inputs, and allows for easy randomization. Perfect for switching around artist mixes at random if using Illu/NoobAI models.

## Random Normal Distribution
![Random Normal Distribution](https://i.imgur.com/S3h4v4A.png)
This node may be useful to someone. All random number generator nodes for ComfyUI seem to only follow a uniform distribution, this instead gives you one with a normal distribution, with the ability to define a min/max and the mean/std.

## SDXL Empty Latent Size Picker v2
![SDXL Empty Latent Size Picker v2](https://i.imgur.com/ejjxpa5.png)
Credits to [cubiq](https://github.com/cubiq) for this. Since the essentials suite is pretty much near-abandoned, I still wanted to make an adjustment to the original emtpy latent size picker node. This has much more resolutions to choose from, a resolution multiplier, and the ability to swap resolutions as just a convenience feature.

## Parameter Nodes

### Model Parameters
![Model Parameters](https://i.imgur.com/xocM4AM.png)
This does require a bit of explaining. ComfyUI cannot output a tuple to a tuple input unless if both tuples are 1:1. If the tuple is different by even one element, it will refuse to work. I don't know why this is, but it means that there needs to be four outputs to this thing because loaders may expect to have a tuple that contains "None" for the model tuple and "Baked VAE" for the VAE tuple, while nodes like rgthree's context and SD Prompt Saver do not want those.

This was about the best implementation I could come up with. It allows for both specifying what model a loader should load, and then saving it to metadata.

### Input Parameters
![Input Parameters](https://i.imgur.com/VMHJ2Xr.png)
Credits to alexopuss and griss for the original code to this. This node was being worked on for the same reason that I wanted to work on model parameters, an actual proper way to get parameters for the sampler and save them to metadata. My issue with the original implementation, were all the unecessary outputs and inputs. That, and this is when I first realized there was an issue with how Comfy manages tuples as inputs. This should output a few extra lines for the schedulers such that it can work with custom KSamplers.

# Schedulers

## Zipf Scheduler
![Zipf Scheduler](https://i.imgur.com/prueISd.png)

This follows [Zipf's Law](https://en.wikipedia.org/wiki/Zipf's_law) as a method to remove noise. It is somewhat similar to the exponential scheduler, though a bit different as it has a heavier tail. Needs around 25-30+ steps for good results.
