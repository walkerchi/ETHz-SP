## Nerf

$$
[\textbf  x, \boldsymbol \theta]\to \text{[R,G,B]}
$$

- $\boldsymbol \theta$ : direction of the  camera
- $\textbf x$ : position, of the point
- $\text{R,G,B}$ : Red, Green, Blue channel 



- nerf doesn't known  what the 3D object is. It only learns the mapping

- for Forward problem, it's not the same since it's not a 3D-2D problem
  $$
  [x,u]  \to [f]
  $$

- The only thing I can used is position encoding






$$
\mathcal L_{\text{phy}} = \Vert Ku - f\Vert
\\
\mathcal L_{\text{data}} = \Vert \text{NN}(f) - u\Vert
\\
\mathcal L = \mathcal L_{\text{phy}} + 1.0 \cdot \mathcal L_{\text{data}}
$$
