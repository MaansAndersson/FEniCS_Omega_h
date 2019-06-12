# Builds a Docker image with FEniCS stable version and Omega_h. The image is at:
#
#    https://quay.io/repository/fenicsproject/stable
#
# Authors:
# Tamara Dancheva <tdancheva@bcamath.org>

FROM quay.io/fenicsproject/stable:latest
MAINTAINER fenics-project <fenics-support@googlegroups.com>

USER root 

# Install omega_h
RUN /bin/bash -c "fenics-version && \
                  git clone https://github.com/tamaradanceva/omega_h.git && \
                  cd omega_h && \
		  git checkout FEniCS && \
	          mkdir build && \
                  cd build && \
		  cmake .. -DCMAKE_CXX_COMPILER:FILEPATH=`which mpicxx` -DCMAKE_C_COMPILER:FILEPATH=`which mpicc` -DBUILD_TESTING:BOOL="ON" -DCMAKE_INSTALL_PREFIX:PATH="/usr/local/" -DOmega_h_EXAMPLES:BOOL="OFF" -DOmega_h_USE_pybind11:BOOL="ON" -DOmega_h_USE_MPI:BOOL="ON" -DOmega_h_USE_OpenMP:BOOL="OFF" -DOmega_h_USE_CUDA:BOOL="OFF" -DOmega_h_ARCH:STRING="" -DOmega_h_USE_DOLFIN:BOOL="ON" && \
                  make install"

ENV PYTHONPATH /usr/local/lib/python3/dist-packages:$PYTHONPATH
USER fenics
WORKDIR $FENICS_HOME
COPY WELCOME $FENICS_HOME/WELCOME

#USER root
