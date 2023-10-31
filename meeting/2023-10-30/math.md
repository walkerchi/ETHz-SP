$$
K_{ij} = \text{MLP}(x_i,x_j)
\\
K_{ij}u_j = f_i
$$

# Forward

$$
x,u,f \in \R^{n\times 2}
$$

# Backward

## Normal Architecture

$$
x,f\in \R^{n\times 2,3}\to u\in \R^{n\times 2,3}
$$

*NodeEdgeGNN*
$$
f\in \R^{n\times2,3},\tilde x\in \R^{e\times 2,3} \to u\in\R^{n\times  2,3}
$$


## Condensed Equivelent Architecture

### b2i

architecture fixed
$$
u\in\R^{b\times 2,3}\to f\in \R^{i\times2,3}
$$
$b+i = n$

### i

the same architecture as  the normal
$$
f\in\R^{i\times2,3}\to u\in\R^{i\times 2,3}
$$
