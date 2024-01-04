

# Dataset 

##  Rectangle

###  Sin2

**force**
$$
p(x) = p_0 \text{sin}(2\pi x  /a)\quad x\in[0, a]
$$

### Sin1

$$
p(x) = p_0\text{sin}(\pi x/a) \quad x\in [0, a]
$$



# Physical Loss

## Strong Form 


$$
\mathcal L_{\text{phy strong}} = \Vert Ku - f \Vert_2
$$


## Weak Form

$$
\mathcal L_{\text{phy weak}} = \left\Vert \mathcal P_{\mathcal V}\left(
\int_\Omega \begin{bmatrix}
\frac{\partial u}{\partial x}&0\\
0&\frac{\partial u}{\partial y}\\
\frac{\partial u}{\partial y}&\frac{\partial u}{\partial x}
\end{bmatrix}_{eid} D_{eij} B_{ebjd}|J|_{e} - f_{ebd}N_{b}|J|_{e}  \text dv\right)
\right\Vert_2
$$

- $D, B$ : vigoit notation
- $\mathcal P_{\mathcal V}$ : assemble matrix , $\mathcal P_{\mathcal V}\in \R^{|\mathcal C|\times b}\to\R^{|\mathcal V|}$
- $(\nabla u)_{ed} = (\nabla N)_{ebd} u_{ebd}$
- $e$ : notation for element 
- $b$ : notation for basis
- $d$ : notation for dimension
- $i,j$ : notation for reduction
