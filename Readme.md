# Example of Coupled Physics Simulation

## Docker environment
A docker environment is provided for ease of use. It can be built from the top-level
directory using

```sh
docker image build . -t mpi:latex
```

It can also be run from the top-level directory for debugging and development

```sh
docker container run --rm -e DISPLAY=host.docker.internal:0 -v ${PWD}:/app -it mpi:latex
```

This mounds the current directory to the landing directory of the container. The `DISPLAY` variable allows xterm to communicate with any
interactive terminals spun up in the container (i.e. `mpirun -n 2 xterm -e gdb path/to/executable`)

**Note:** on a Mac, need to additionally start Xserver and use `xhost +localhost`

## Build Documentation

Documentation is built through a podman container. First build the image with

```sh
podman image build . -f Containerfile.latex -t coupled-physics:latex
```

Generate the documentation with

```sh
podman container run --rm -v ./Documentation:/app -it coupled-physics:latex
```

Clean documentation

```sh
podman container run --rm -v ./Documentation:/app -it coupled-physics:latex make clean
```
