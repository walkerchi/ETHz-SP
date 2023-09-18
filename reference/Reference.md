# Introduction

## In FEM View

The **strong form** :  $\nabla\cdot \sigma(\varepsilon) + f = \rho\frac{\partial^2 u}{\partial t^2}$

The **weak form** : $\int_\Omega (\nabla\cdot \sigma)\cdot v+b\cdot v\text d\Omega = \rho\int_\Omega \frac{\partial^2 u }{\partial t^2}\cdot v\text d \Omega\Leftrightarrow \int_\Omega\sigma:\nabla v + b\cdot v\text d\Omega-\int_{\partial \Omega}(\sigma\cdot n)\cdot v\text dS=\rho\int_\Omega\frac{\partial^2 u}{\partial t^2}\cdot v\text d\Omega$

- $\sigma:\nabla v = \nabla v^\top \cdot \sigma \cdot \nabla v\quad \sigma\in \R^{d\times d},\nabla v\in \R^d$

- $\varepsilon$ : strain, $\varepsilon = \nabla u$

- $b$ : body force act on each element unit, e.g, Gravity

- $\sigma(\varepsilon)$ : stress, a non linear function of strain $\varepsilon$

  ![img](Reference.assets/Stress_strain_ductile.svg?darksrc=invert#center#400px)

  -  Lame's approximation : $\sigma=\lambda \text{Tr}(\varepsilon)I+2\mu\varepsilon\quad λ=\frac{\nu E}{(1+\nu)(1-2\nu)}, \mu = \frac{E}{1+2\nu}$

- no acceleration assumption $\frac{\partial ^2 u }{\partial t^2} = 0 $ , no gravity assumption $b = 0$

- bilinear form : $a(u,v) =  \int_\Omega \sigma(\varepsilon) :\nabla v + b \cdot v\text d\Omega$ and linear form : $\ell(v) = \int_{\partial \Omega}(\sigma\cdot n)\cdot \text dS$

- Galerkin discretilization : $K_{ij} = \int_\Omega \sigma(\sum u_i\phi_i):\nabla \psi_j + b\cdot \psi_j\text d\Omega$ where $\phi_i=N_i$ trial space shape function, $\psi_j=N_j$ test space shape function

### Linear Case

When Dirichlet Boundary Condition applied to mesh $\mathcal M$, $u_B,K,f\to u_I$


$$
\begin{bmatrix}K_{II}&K_{IB}\\K_{IB}^\top&K_{BB}\end{bmatrix}
\begin{bmatrix}u_I\\u_B\end{bmatrix}=
\begin{bmatrix}f_I\\f_B\end{bmatrix}
\to 
K_{II}x_I = f_I - K_{IB}x_B
$$

- $A_{\{II,BB,IB\}}$ : Galerkin matrix for interior/boundary/interior-boundary connection
  - element matrix : $K_e = \int_\Omega B_e^\top D_eB_e \text d\Omega$
  - $B$ :  Strain-Displacement Matrix, $\varepsilon = B u = \frac{1}{2}(\nabla u + \nabla u^\top)$
    - Triangle Element : $B = \begin{bmatrix}\frac{\partial N_1}{\partial x}&0&\frac{\partial N_2}{\partial x}&0&\frac{\partial N_3}{\partial x}&0\\ 0 & \frac{\partial N_1}{\partial y}&0&\frac{\partial N_2}{\partial y}&0&\frac{\partial N_3}{\partial y}\\\frac{\partial N_1}{\partial y}&\frac{\partial N_1}{\partial x}&\frac{\partial N_2}{\partial y}&\frac{\partial N_2}{\partial x}&\frac{\partial N_3}{\partial y}&\frac{\partial N_3}{\partial x}\end{bmatrix}\quad \varepsilon = \begin{bmatrix}\varepsilon_x\\\varepsilon_y\\  2\varepsilon_{xy}\end{bmatrix}$
  - $D$ : Material's Constitutive (Elasticity) Matrix, $\sigma = D\varepsilon=DBu$
    - Triangle Element : $D = \frac{E}{(1+\nu)(1-2\nu)}\begin{bmatrix}1-\nu & \nu & 0 \\ \nu & 1-\nu & 0 \\ 0 & 0 & \frac{1-2\nu}{2}\end{bmatrix}$
- $u_{\{I,B\}}$ : displacement for interior/boundary nodes
- $f_{\{I,B\}}$ : applied force for interior/ boundary nodes

## In NN View

### Linear Case


$$
Ku = f \Leftrightarrow x = \text{GNN}(A,x)
$$

considering Dirichlet Boundary Condition
$$
\begin{aligned}
K_{II}u_I = f_{I}-K_{IB}u_B\Leftrightarrow u_I = \text{GNN}(K_{II},f_I - K_{IB}u_B)
\end{aligned}
$$

## Purpose

1. End2End GNN framework applied to Dirichlet Boundary Condition
2. (Faster speed, larger scale)
3. Better precision
4. (Non-Lame's approximation)
5. survey

# Background

## Physics GNN



## Boundary GNN


### Fully Connect

![img](Reference.assets/boundary_gnn.jpeg?darksrc=invert#center#400px)![img](Reference.assets/boundary_gnn_nfeat.jpeg?darksrc=invert#center#400px)![img](Reference.assets/boundary_gnn_example.jpeg?darksrc=invert#center#400px)
$$
(v_i,v_j)\in\hat{\mathcal E}\quad \forall v_i\in \mathcal M_I\quad\forall v_j\in \mathcal M_B
$$

- interior nodes and boundary nodes fully connected
- element as node in graph

> Xingyu Fu, Fengfeng Zhou, Dheeraj Peddireddy, Zhengyang Kang, Martin Byung-Guk Jun, Vaneet Aggarwal, An finite element analysis surrogate model with boundary oriented graph embedding approach for rapid design, *Journal of Computational Design and Engineering*, Volume 10, Issue 3, June 2023, Pages 1026–1046, https://doi.org/10.1093/jcde/qwad025



### Physical Informed

![img](Reference.assets/StructureGNN-E.jpg?darksrc=invert#center#800px)

- assumption : $\Vert f_{I} + f_B\Vert = 0$
- node as node in graph

> Ling-Han Song, Chen Wang, Jian-Sheng Fan, Hong-Ming Luformat_quoteCITE, Elastic structural analysis based on graph neural network without labeled data, *Computer-Aided Civil and Infrastructure Engineering*, Volume 38, Issue 10, Pages1237-1399, https://onlinelibrary.wiley.com/doi/epdf/10.1111/mice.12944

## Graph Pooling

### K-Hop



> Yadi Cao, Menglei Chai, Minchen Li, Chenfanfu Jiang, Efficient Learning of Mesh-Based Physical Simulation with BSMS-GNN,*arXiv*:2210.02573, github, [GitHub - Eydcao/BSMS-GNN](https://github.com/Eydcao/BSMS-GNN)



### Learnable Pooling

### Rasterization Pooling

# Methodology

# Experiments



## Strain/Stress Precision

 

## Speed/Memory Comparation



## Case Study





## Topology Optimization

$$
z = \underset{z}{\text{argmin}}\sum_{e=1}^N (z_e)^p u_e^\top A_eu_e\quad z_e > 0 
$$

