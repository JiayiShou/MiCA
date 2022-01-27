# MiCA
Mixture of Autoencoder

The goal of this project is to design a protein classifier inpired by the mixture of experts concept. 
In this case, the experts are autoencoders. The toy example includes a connected stacked autoencoder, 
a variational autoencoder, and a denoised autoencoder as experts. In addition, there will be a gating 
function performing soft partitions for each autoencoder. The idea is that if autoencoders became experts
in their own domain, then they will perform better classification.

