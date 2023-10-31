$$
K^{ij} = \int _\Omega \C_{ijkl} \nabla N_l^j\nabla N_k^i\text dv
$$

$$
K^{ij}=\sum_{m}\phi_m\C(\xi_m)_{ijkl}\nabla N^j(\xi_m)_l\nabla N^i(\xi_m)
$$

$$
\begin{aligned}
&\mathcal M_\xi(\{x^i|x^i\in\mathcal C\})\in \R^{b\times d\to \R^{|\phi|\times d}}
\\
&\mathcal M_{\nabla\phi}(\xi) \in \R^{|\phi|\times d}\to \R^{b\times d}
\\
&\mathcal M_\C(\xi)\in \R^{d}\to \R^{d\times d\times b\times b}
\end{aligned}
$$


$$
K^{ij}_{\text{local}} = \frac{\phi \mathcal M_{\mathbb C}\left(\xi_{\mathcal M_m}\right)_{ijkl}\mathcal M_{\nabla N}(\xi_{\mathcal M})^j_lM_{\nabla N}(\xi_{\mathcal M})^i_k}{\sum\phi}
$$

$$
K_{\text{global}}^{nkl} = \mathcal P_{\mathcal E}^{nhij} K_{\text{local}}^{hklij}
$$

$$
K_{\text{global}}\overset{\text{bsr matrix}}{\rightarrow}\hat K_{\text{global}}
$$

- $N$ : basis function
- $i,j$ : basis function notation
- $k,l$ : dimension notation
- $m$ : quadrature notation
- $n$ : edge notation
- $h$ : element notation
- $\phi$ : quadrature weight
- $\xi$ : quadrature point $\xi_m\in \R^d$
- $b$ : number of basis
- $d$ : number of dimension
- $c$ : number of cell/elements
- $\mathcal C$ : cell/element, which has $b$ basis
- $\mathcal E$ : edges in the graph representation
- $\mathcal V$ : number of points/vertex/basis of  the graph representation
- $K_{\text{local}}$ : local stiffness/Galerkin tensor, $K_{\text{local}}\in \R^{c\times b\times b\times d\times d}$
- $K_{\text{global}}$ : global stiffness/Galerkin tensor, $K_{\text{global}}\in \R^{|\mathcal E|\times  d\times d}$
- $\hat K_{\text{global}}$ : global stiffness/Galerkin matrix, $\hat K_{\text{global}}\in \R_{\text{sparse}}^{(|\mathcal V|\times d)\times(|\mathcal V|\times d)}$
- $\mathcal P_{\mathcal E}$ : projection tensor from $K_{\text{local}}$ to $K_{\text{global}}$, $\mathcal P_{\mathcal E} \in \R_{\text{sparse}}^{|\mathcal E|\times c\times b\times b}$
