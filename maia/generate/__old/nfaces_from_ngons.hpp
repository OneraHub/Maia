#pragma once


#include "mpi.h"
namespace cgns { struct tree; } // Fwd decl


namespace maia {


auto
add_nfaces(cgns::tree& z, MPI_Comm comm) -> void;


} // maia
