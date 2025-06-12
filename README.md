# comfyui-lopi999-nodes
In general, this is a sort of "general purpose" node suite, with some unique nodes and some modified nodes. This suite adds two schedulers and four samplers to the default Comfy scheduler and sampler lists. The samplers added are extsig modifications to the Euler samplers, which cause them to spend more time in lower sigmas, increasing detail. The two schedulers are covered in more depth later in this README.

# Nodes

## Image Nodes
![Image Nodes](https://files.catbox.moe/tjfcem.png)

These two nodes are based on the burn and HSV treatment functions provided by Krita. Good for some post-processing after generating an image.

## Random Nodes

### Random Boolean
![Random Boolean](https://files.catbox.moe/k0ydtt.png)

Straightforward and simple, just generates a True or a False randomly each time. Also contains int outputs for some extended modularity.

### Random Normal Distribution
![Random Normal Distribution](https://files.catbox.moe/630bjv.png)

All current random number generators for Comfy really only give uniform distributions. Why not normal distributions as well? Useful for a more "controlled" randomness in large batches.

### Random SDXL Latent Size
![Random SDXL Latent Size](https://i.imgur.com/n1xiaKh.png)

This node is sort of the one that started this collection of nodes. It is a fairly straightforward node based heavily on the Empty Latent Picker node from [ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials). Most of the functions are pretty self-explanitory. Landscape and portrait resolutions can either be included or excluded, and one can define what the minimum landscape and the maximum landscape resolutions should be.

## Schedulers
![Schedulers](https://files.catbox.moe/e6o8eq.png)

Adds two schedulers, Zipf Linear and Zeta. Zipf Linear is really designed to be good for a "flat" look, as it drops heavily down towards low sigmas after only a few steps. Zeta is meant to be a sort of "alternative" to Beta, they work on different distribution types, but overall Zeta appears to have good color depth.

Both schedulers have a linear drift that goes from high to low x.

## String Nodes

### Advanced Text Switch
![Advanced Text Switch](https://i.imgur.com/P7x7WHT.png)

Take the text switch node, and use built-in multiline textboxes instead of text inputs, allow for easy randomization and concatenation, and you get the Advanced Text Switch node. Technically should have code to hide textboxes that aren't in use but I guess I wasn't able to get the whale to properly code that in. Otherwise, it is perfectly functional, helps trim down on node counts when using multiple text inputs, and allows for easy randomization. Perfect for switching around artist mixes at random if using Illu/NoobAI models.

### Concatenate With Prefix
![Concatenate With Prefix](https://files.catbox.moe/j8hsp0.png)

Most concatenation nodes only let you specify a delimiter. Why not a prefix as well? Effortlessly prefixes nodes as well as delimits, with the added ability to have a dynamic counter for the prefix.

### List Wildcard
![List Wildcard](https://files.catbox.moe/z02kti.png)

Intended to be a dead-simple wildcard node that just takes a separator and makes a list out of a given string, and randomly selects an item from the list.

### Parameters to String
![Parameters to String](https://files.catbox.moe/ur8tdt.png)

Takes some sampler parameters, and converts it into a string. Useful for the extra info input for the Prompt Saver Node if using more than one ksampler.

## Etc.

### Model Parameters
![Model Parameters](https://i.imgur.com/xocM4AM.png)
This does require a bit of explaining. ComfyUI cannot output a tuple to a tuple input unless if both tuples are 1:1. If the tuple is different by even one element, it will refuse to work. I don't know why this is, but it means that there needs to be four outputs to this thing because loaders may expect to have a tuple that contains "None" for the model tuple and "Baked VAE" for the VAE tuple, while nodes like rgthree's context and SD Prompt Saver do not want those.

This was about the best implementation I could come up with. It allows for both specifying what model a loader should load, and then saving it to metadata.

### SDXL Empty Latent Size Picker v2
![SDXL Empty Latent Size Picker v2](https://i.imgur.com/ejjxpa5.png)
Credits to [cubiq](https://github.com/cubiq) for this. Since the essentials suite is pretty much near-abandoned, I still wanted to make an adjustment to the original emtpy latent size picker node. This has much more resolutions to choose from, a resolution multiplier, and the ability to swap resolutions as just a convenience feature.

### Token Counter
![Token Counter](https://files.catbox.moe/e3vspl.png)
Credits to pamparamm for the original code provided here. The current code for ppm's CLIP Token Counter has issues with... just giving a simple, straightforward output. This is based on the old code for that node, with two very slight modifications:
1. String is now a forced input instead of a multiline string being built into the node
2. The node can now output an int as well as a string, useful for manipulation if needed.


