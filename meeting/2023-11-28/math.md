$$
\begin{bmatrix}
A_{II}& A_{IB}\\
A_{IB}^\top & A_{BB}
\end{bmatrix}
\begin{bmatrix}
u_I \\
u_B 
\end{bmatrix}
= 
\begin{bmatrix}
f_I \\ f_B
\end{bmatrix}
$$

$$
\begin{cases}
A_{II} u_I = f_I - A_{IB}u_B \\
A_{IB}^\top u_I + A_{BB}u_B = f_B
\end{cases}
$$

known : $A, u_B, f_I$

unknown : $u_I, f_B$



when `train_ratio=0.0`
$$
\begin{cases}
\mathcal L_{\text{train}} = \mathcal L_{\text{phy}} =\begin{cases}
\Vert Ku - f \Vert_2 & \text{strong pinn}\\
 \left\Vert \mathcal P_{\mathcal V}\left(
\int_\Omega \begin{bmatrix}
\frac{\partial u}{\partial x}&0\\
0&\frac{\partial u}{\partial y}\\
\frac{\partial u}{\partial y}&\frac{\partial u}{\partial x}
\end{bmatrix}_{eid} D_{eij} B_{ebjd}|J|_{e} - f_{ebd}N_{b}|J|_{e}  \text dv\right)
\right\Vert_2 & \text{weak pinn}
\end{cases}
\\
\mathcal L_{\text{valid}} = \mathcal L_{\text{data}} = \Vert \text{NN}(u_B,f_I) - u_I \Vert_2
\end{cases}
$$




GAT : 

- $n_{\text{heat}}=4$
- $n_{\text{layers}}=3$
- `train_ratio=0.0`



GCN : 

- $n_{\text{layers}}=3$
- `train_ratio=0.0`



SIGN:

- $n_{\text{hops}}$ = 8
- $n_{\text{layers}}=3$
- `train_ratio=0.0`

