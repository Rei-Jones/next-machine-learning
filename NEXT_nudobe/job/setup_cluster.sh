# Set the path to the Geant4 Installation
export G4INSTALL=~/geant4-install
export PATH=$G4INSTALL/bin:$PATH;
export PATH=$G4INSTALL/include:$PATH;
export LD_LIBRARY_PATH=$G4INSTALL/lib:$LD_LIBRARY_PATH;

source $G4INSTALL/bin/geant4.sh;


export HDF5_PATH=/usr/lib/x86_64-linux-gnu/hdf5/serial/
export HDF5_LIB=${HDF5_PATH}/lib/;
export HDF5_INC=${HDF5_PATH}/include/;
export LD_LIBRARY_PATH=$HDF5_LIB:$LD_LIBRARY_PATH;
export HDF5_DIR=$HDF5_PATH

# Add this GSL stuff as compilation failing
export GSL_PATH=/usr/local;
export PATH=$GSL_PATH/bin:$PATH;
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GSL_PATH/lib;
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu

export PATH=$PATH:~/NEXT/nexus/build

echo "Setup Nexus is complete!!"
