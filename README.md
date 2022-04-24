# Code and examples for Master Thesis

## Implementations
Implementations of VQVAE and PixelSNAIL found in models directory.

VQVAE: Hierarchical Quantized Autoencoders [https://arxiv.org/abs/2002.08111]
PixelSNAIL: An Improved Autoregressive Generative Model [https://arxiv.org/abs/1712.09763]

## Examples

### Reconstructions with Hierarchical VQVAE
16 256x256 frames in 10 FPS decoded and encoded.
Assuming 8-bit color channels, 3x16x256x256x8 is encoded into 4x32x32 + 8x32x32, which is a 98.8% reduction.

<p float="left">
  <img src="./examples/vqvae_reconstructions/1.gif" width="450" />
  <img src="./examples/vqvae_reconstructions/2.gif" width="450" /> 
</p>
<p float="left">
  <img src="./examples/vqvae_reconstructions/3.gif" width="450" />
  <img src="./examples/vqvae_reconstructions/4.gif" width="450" /> 
</p>

#### Reconstruction using only top/bottom encoding

The lower dimensioned top encoding takes care of more general and global features, like coloring. The higher dimensioned bottom encoding takes care of more detailed features. Here, the input (left) is decoded first using only the top encoding (middle), then by the bottom encoding (right).

<p float="left">
  <img src="./examples/input_top_bot_separate/synced.gif" width="900" />
</p>

### PixelSNAIL
#### Top PixelSNAIL
Example showing ancestral sampling conditioned on 8 frames. 8 new frames are generated. Left is decoded from original encoding, middle is decoded from the generated top encoding and right is encoded from the generated top encoding and the bottom encoding (not generated!)

<p float="left">
  <img src="./examples/pixelsnail/top_snail/50msyncedd.gif" width="900" />
</p>
