# Learning Structural Dynamics with Graph Neural Networks

## Background

### Graph Element Network

1. dynamic meshing with fixed nodal number

### Neural Process

1. gaussian process-like NN (basis->query)

### Physics Informed-Graph  Convolution Network

1. inverse solution
2. basis function same as FEM



## Thoughts

1. All the nodes within the support should be neighbors

   -> only triangle/tetrahedron can be directly used

   

2. **important** dynamic meshing with varying nodal number(the less the better compared to FEM)

   

3. relative position as aggregation attention

   

4. depth of GNN >= diam(mesh) -> maybe a UNet structure?

   1. depth of UNet depends on the diam(mesh)

   2. pooling and unpooling should  be different for different scale(renormalization group)

      

5. FEM mesh is quiet different from social network, no neighbor explosion( six  degree of seperation)

   

6. potential  advantage  GNN over FEM: less nodes, less memory, faster solution

   

7. if a graph is as big as possible -> high dimension manifold

   more neighbors -> higher curvature, non-differentiable

   

   

## Questions

1. Framework selection PyG(more nn)/DGL(general)/pure Pytorch(flexible, less installation)

   

1. input/output, continuous/discrete, whether physics informed, 