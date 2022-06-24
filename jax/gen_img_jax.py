import jax.numpy as jnp
import matplotlib.pyplot as plt


def gen_img(coord, args_dict):
    
    n_atoms = coord.shape[1]
    norm = 1 / (2 * jnp.pi * args_dict["SIGMA"]**2 * n_atoms)

    grid_min = -args_dict["PIXEL_SIZE"] * (args_dict["N_PIXELS"] - 1)*0.5
    grid_max = args_dict["PIXEL_SIZE"] * (args_dict["N_PIXELS"] - 1)*0.5 + args_dict["PIXEL_SIZE"]

    grid = jnp.arange(grid_min, grid_max, args_dict["PIXEL_SIZE"])

    gauss = jnp.exp( -0.5 * ( ((grid[:,None] - coord[0,:]) / args_dict["SIGMA"])**2) )[:,None] * jnp.exp( -0.5 * ( ((grid[:,None] - coord[1,:]) / args_dict["SIGMA"])**2) )

    image = gauss.sum(axis=2) * norm

    return image


def main():

    args_dict = {"PIXEL_SIZE": 4, "N_PIXELS": 32, "SIGMA": 1.0}

    coord = jnp.array([[0.0], [0.0], [0.0]])
    image = gen_img(coord, args_dict)

    plt.imshow(image.T, origin="lower")
    plt.show()


if __name__ == "__main__":

    main()