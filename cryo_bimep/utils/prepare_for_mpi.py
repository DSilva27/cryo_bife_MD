"Prepare variables to run with mpi"


def prep_for_mpi(path, rank, world_size):

    if world_size == 1:
        return path
    
    else:
        if rank == 0:
            path_rank = path[0:2]

        elif rank == world_size - 1:
            path_rank = path[-2:]

        else:
            path_rank = path[rank + 1:rank + 2]

        return path_rank
