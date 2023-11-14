from dataset import Truss, Triangle, Tetra

import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="rectangle", choices=Truss.NAMES + Triangle.NAMES + Tetra.NAMES)

args = parser.parse_args()

def main(args):
    if args.dataset in Truss.NAMES:
        truss = Truss(name=args.dataset)
        u = truss.solve()
        r = truss.compute_residual(u, mse=True)
        print(f"residual: {r}")
        truss.plot(ux = u[:, 0], uy = u[:, 1])
    elif args.dataset in Triangle.NAMES:
        triangle = Triangle(name=args.dataset)
        u = triangle.solve()
        r = triangle.compute_residual(u, mse=True)
        print(f"residual: {r}")
        triangle.plot(ux = u[:,0], uy = u[:,1])
    elif args.dataset in Tetra.NAMES:
        tetra = Tetra(name=args.dataset)
        u = tetra.solve()
        r = tetra.compute_residual(u, mse=True)
        print(f"residual: {r}")
        tetra.plot(ux = u[:,0], uy = u[:,1], uz = u[:,2])
    else:
        raise NotImplementedError



if __name__ == '__main__':
    main(args)