$$
4\times 7 \times(24 + 6) = 840
$$

$$
\begin{bmatrix}
\begin{bmatrix}
0 & 0\\
&6&7&8\\

\end{bmatrix}
\end{bmatrix}

\begin{bmatrix}
\begin{bmatrix}
k_{11}&K_{12}&k_{13}&k_{14}\\
k_{21}&k_{22}&k_{23}&k_{24}\\
k_{31}&k_{32}&k_{33}&k_{34}\\
k_{41}&k_{42}&k_{43}&k_{44}
\end{bmatrix}\\
\begin{bmatrix}
k_{11}&K_{12}&k_{13}&k_{14}\\
k_{21}&k_{22}&k_{23}&k_{24}\\
k_{31}&k_{32}&k_{33}&k_{34}\\
k_{41}&k_{42}&k_{43}&k_{44}
\end{bmatrix}
\end{bmatrix}
$$




$$
K,f\rightarrow u
$$

$$
f,u\rightarrow K
$$


## GCN

$$
H^{l+1} = \sigma(\underbrace{D^{-\frac{1}{2}}(A+I)D^{-\frac{1}{2}}}_{\mathcal L
} H^{l}W^l + b^l)
$$

## GAT

$$
H^{l+1}  = \sigma\left(\sum_{j\in\mathcal N_i\cup\{i\}}\alpha_{ij} \textbf Wh_j\right)
\\
\alpha_{ij}= \frac{\text{exp}(\text{LeakyReLU}(\textbf a^\top [\textbf Wh_i\Vert \textbf  Wh_j]))}{\sum_{k\in \mathcal N_i\cup \{i\}}\text{exp}(\text{LeakyReLU}(\textbf a^\top[\textbf Wh_i \Vert \textbf Wh_k]))}
$$



## SIGN

$$
H = \text{MLP}(\Vert_{i=0}^{n} \text{MLP}_i(L^i X))
$$



## Node-Edge GNN

$$
\begin{align}\label{eq:node-edge-gnn}
       H_{\mathcal E}^{l+1}&\gets \sigma\left(W_{\mathcal E}\left([H_{\mathcal V,u}^l\Vert H_{\mathcal V,v}^l \Vert H^l_{\mathcal E}]\right) + b_{\mathcal E}\right)\nonumber
       \\
       H_{\mathcal V,i}^{l+1}&\gets \sigma\left(W_{\mathcal V}\frac{1}{\vert \mathcal N_i \cup \{i\}\vert} \sum_{j\in\mathcal N_i}H_{\mathcal E,ij}^{l+1} + b_{\mathcal V}\right)
    \end{align}
$$

## Static Condensation

$$
u_i = \text{GNN}_{\theta_1}(A_{ii},f_i-K_{ei}u_e,x_i)
$$

## NN Condensation

$$
u_i = \text{GNN}_{\theta_1}(A_{ii},f_i+\text{B-GNN}_{\theta_2}(A_{ei},u_e),x_i)
$$

## Edge-GNN

$$
K_e = \text{MLP}([x_u||x_v])
$$

## B-GNN Layer

$$

$$




$$
\mathcal P_{\mathcal E}\in \R^{14\times 18}
$$



$$
\mathcal V = \begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix} + \mathcal N(0, 0.4)
$$

$$
\mathcal V = \begin{bmatrix}
0 & 0 & 1 & 1 \\
0 & 1 & 0 & 1
\end{bmatrix} + \mathcal N(0, 0.4)
$$

