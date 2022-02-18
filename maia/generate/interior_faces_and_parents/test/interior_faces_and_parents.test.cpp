#include "std_e/unit_test/doctest_pybind_mpi.hpp"

#include "maia/utils/yaml/parse_yaml_cgns.hpp"
#include "maia/utils/mesh_dir.hpp"
#include "std_e/utils/file.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "maia/sids/element_sections.hpp"

#include "maia/generate/interior_faces_and_parents/interior_faces_and_parents.hpp"
#include "maia/generate/interior_faces_and_parents/element_faces_and_parents.hpp"
#include "std_e/parallel/mpi/base.hpp"
#include "std_e/multi_array/utils.hpp"
#include "std_e/log.hpp"


using namespace cgns;


PYBIND_MPI_TEST_CASE("generate_interior_faces_and_parents - seq",1) {
  std::string yaml_tree = std_e::file_to_string(maia::mesh_dir+"hex_2_prism_2_dist_01.yaml");
  tree b = maia::to_node(yaml_tree);

  tree& z = cgns::get_node_by_name(b,"Zone");

  maia::generate_interior_faces_and_parents(z,test_comm);
  // tri ext
  auto parent_tri_ext     = cgns::get_node_value_by_matching<I4,2>(b,"Zone/Tris/ParentElements");
  auto parent_pos_tri_ext = cgns::get_node_value_by_matching<I4,2>(b,"Zone/Tris/ParentElementsPosition");
  CHECK( parent_tri_ext.extent() == std_e::multi_index<I8,2>{2,2} );
  CHECK( parent_tri_ext     == cgns::md_array<I4,2>{{17,0},{18,0}} );
  CHECK( parent_pos_tri_ext == cgns::md_array<I4,2>{{ 4,0},{ 5,0}} );

  // tri in
  auto elt_type_tri_in   = cgns::get_node_value_by_matching<I4>(b,"Zone/TRI_3_interior")[0];
  auto range_tri_in      = cgns::get_node_value_by_matching<I4>(b,"Zone/TRI_3_interior/ElementRange");
  auto connec_tri_in     = cgns::get_node_value_by_matching<I4>(b,"Zone/TRI_3_interior/ElementConnectivity");
  auto parent_tri_in     = cgns::get_node_value_by_matching<I4,2>(b,"Zone/TRI_3_interior/ParentElements");
  auto parent_pos_tri_in = cgns::get_node_value_by_matching<I4,2>(b,"Zone/TRI_3_interior/ParentElementsPosition");

  CHECK( elt_type_tri_in == (I4)cgns::TRI_3 );
  CHECK( range_tri_in == std::vector<I4>{15,15} );
  CHECK( connec_tri_in == std::vector<I4>{7,8,10} );
  CHECK( parent_tri_in.extent() == std_e::multi_index<I8,2>{1,2} );
  CHECK( parent_tri_in     == cgns::md_array<I4,2>{{17,18}} );
  CHECK( parent_pos_tri_in == cgns::md_array<I4,2>{{ 5, 4}} );

  // quad ext
  auto parent_quad_ext     = cgns::get_node_value_by_matching<I4,2>(b,"Zone/Quads/ParentElements");
  auto parent_pos_quad_ext = cgns::get_node_value_by_matching<I4,2>(b,"Zone/Quads/ParentElementsPosition");
  CHECK( parent_quad_ext.extent() == std_e::multi_index<I8,2>{12,2} );
  CHECK( parent_quad_ext     == cgns::md_array<I4,2>{{15,0},{16,0},{17,0},{18,0},{15,0},{17,0},{16,0},{18,0},{15,0},{16,0},{15,0},{16,0}} );
  CHECK( parent_pos_quad_ext == cgns::md_array<I4,2>{{ 5,0},{ 5,0},{ 2,0},{ 2,0},{ 2,0},{ 1,0},{ 2,0},{ 1,0},{ 4,0},{ 4,0},{ 1,0},{ 6,0}} );

  // quad in
  auto elt_type_quad_in   = cgns::get_node_value_by_matching<I4>(b,"Zone/QUAD_4_interior")[0];
  auto range_quad_in      = cgns::get_node_value_by_matching<I4>(b,"Zone/QUAD_4_interior/ElementRange");
  auto connec_quad_in     = cgns::get_node_value_by_matching<I4>(b,"Zone/QUAD_4_interior/ElementConnectivity");
  auto parent_quad_in     = cgns::get_node_value_by_matching<I4,2>(b,"Zone/QUAD_4_interior/ParentElements");
  auto parent_pos_quad_in = cgns::get_node_value_by_matching<I4,2>(b,"Zone/QUAD_4_interior/ParentElementsPosition");

  CHECK( elt_type_quad_in == (I4)cgns::QUAD_4 );
  CHECK( range_quad_in == std::vector<I4>{16,18} );
  CHECK( connec_quad_in == std::vector<I4>{2,7,10,5, 6,7,10,9, 7,10,15,12} );
  CHECK( parent_quad_in.extent() == std_e::multi_index<I8,2>{3,2} );
  CHECK( parent_quad_in     == cgns::md_array<I4,2>{{17,15},{15,16},{16,18}} );
  CHECK( parent_pos_quad_in == cgns::md_array<I4,2>{{ 3, 3},{ 6, 1},{ 3, 3}} );

  // hex
  auto hex_cell_face = cgns::get_node_value_by_matching<I4>(b,"Zone/Hexas/CellFace");
  CHECK( hex_cell_face == std::vector<I4>{11,5,16,9,1,17,  17,7,18,10,2,12} );
  // prisms
  auto prism_cell_face = cgns::get_node_value_by_matching<I4>(b,"Zone/Prisms/CellFace");
  CHECK( prism_cell_face == std::vector<I4>{6,3,16,13,15, 8,4,18,15,14} );
}


PYBIND_MPI_TEST_CASE("generate_interior_faces_and_parents",2) {
  int rk = std_e::rank(test_comm);
  std::string yaml_tree = std_e::file_to_string(maia::mesh_dir+"hex_2_prism_2_dist_"+std::to_string(rk)+".yaml");
  tree b = maia::to_node(yaml_tree);

  tree& z = cgns::get_node_by_name(b,"Zone");

  maia::generate_interior_faces_and_parents(z,test_comm);
  // tri ext
  auto parent_tri_ext     = cgns::get_node_value_by_matching<I4,2>(b,"Zone/Tris/ParentElements");
  auto parent_pos_tri_ext = cgns::get_node_value_by_matching<I4,2>(b,"Zone/Tris/ParentElementsPosition");
  CHECK( parent_tri_ext.extent() == std_e::multi_index<I8,2>{1,2} );
  MPI_CHECK(0, parent_tri_ext == cgns::md_array<I4,2>{{17,0}} );
  MPI_CHECK(1, parent_tri_ext == cgns::md_array<I4,2>{{18,0}} );
  MPI_CHECK(0, parent_pos_tri_ext == cgns::md_array<I4,2>{{ 4,0}       } );
  MPI_CHECK(1, parent_pos_tri_ext == cgns::md_array<I4,2>{       { 5,0}} );

  // tri in
  auto elt_type_tri_in   = cgns::get_node_value_by_matching<I4>(b,"Zone/TRI_3_interior")[0];
  auto range_tri_in      = cgns::get_node_value_by_matching<I4>(b,"Zone/TRI_3_interior/ElementRange");
  auto connec_tri_in     = cgns::get_node_value_by_matching<I4>(b,"Zone/TRI_3_interior/ElementConnectivity");
  auto parent_tri_in     = cgns::get_node_value_by_matching<I4,2>(b,"Zone/TRI_3_interior/ParentElements");
  auto parent_pos_tri_in = cgns::get_node_value_by_matching<I4,2>(b,"Zone/TRI_3_interior/ParentElementsPosition");

  CHECK( elt_type_tri_in == (I4)cgns::TRI_3 );
  CHECK( range_tri_in == std::vector<I4>{15,15} );
  ELOG(connec_tri_in);
  ELOG(parent_tri_in);
  ELOG(parent_pos_tri_in);
  //MPI_CHECK(0, connec_tri_in == std::vector<I4>{7,8,10} );
  //MPI_CHECK(1, connec_tri_in == std::vector<I4>{} );
  //MPI_CHECK(0, parent_tri_in.extent() == std_e::multi_index<I8,2>{1,2} );
  //MPI_CHECK(1, parent_tri_in.extent() == std_e::multi_index<I8,2>{0,2} );
  //MPI_CHECK(0, parent_tri_in == cgns::md_array<I4,2>{{17,18}} );
  //MPI_CHECK(1, parent_tri_in == cgns::md_array<I4,2>(0,2) );
  //MPI_CHECK(0, parent_pos_tri_in == cgns::md_array<I4,2>{{ 5, 4}} );
  //MPI_CHECK(1, parent_pos_tri_in == cgns::md_array<I4,2>(0,2) );
  MPI_CHECK(0, connec_tri_in == std::vector<I4>{} );
  MPI_CHECK(1, connec_tri_in == std::vector<I4>{7,8,10} );
  MPI_CHECK(0, parent_tri_in.extent() == std_e::multi_index<I8,2>{0,2} );
  MPI_CHECK(1, parent_tri_in.extent() == std_e::multi_index<I8,2>{1,2} );
  MPI_CHECK(0, parent_tri_in == cgns::md_array<I4,2>(0,2) );
  MPI_CHECK(1, parent_tri_in == cgns::md_array<I4,2>{{17,18}} );
  MPI_CHECK(0, parent_pos_tri_in == cgns::md_array<I4,2>(0,2) );
  MPI_CHECK(1, parent_pos_tri_in == cgns::md_array<I4,2>{{ 5, 4}} );

  // quad ext
  auto parent_quad_ext     = cgns::get_node_value_by_matching<I4,2>(b,"Zone/Quads/ParentElements");
  auto parent_pos_quad_ext = cgns::get_node_value_by_matching<I4,2>(b,"Zone/Quads/ParentElementsPosition");
  CHECK( parent_quad_ext.extent() == std_e::multi_index<I8,2>{6,2} );
  MPI_CHECK(0, parent_quad_ext == cgns::md_array<I4,2>{{15,0},{16,0},{17,0},{18,0},{15,0},{17,0}                                          } );
  MPI_CHECK(1, parent_quad_ext == cgns::md_array<I4,2>{                                          {16,0},{18,0},{15,0},{16,0},{15,0},{16,0}} );
  MPI_CHECK(0, parent_pos_quad_ext == cgns::md_array<I4,2>{{ 5,0},{ 5,0},{ 2,0},{ 2,0},{ 2,0},{ 1,0}                                          } );
  MPI_CHECK(1, parent_pos_quad_ext == cgns::md_array<I4,2>{                                          { 2,0},{ 1,0},{ 4,0},{ 4,0},{ 1,0},{ 6,0}} );

  // quad in
  auto elt_type_quad_in   = cgns::get_node_value_by_matching<I4>(b,"Zone/QUAD_4_interior")[0];
  auto range_quad_in      = cgns::get_node_value_by_matching<I4>(b,"Zone/QUAD_4_interior/ElementRange");
  auto connec_quad_in     = cgns::get_node_value_by_matching<I4>(b,"Zone/QUAD_4_interior/ElementConnectivity");
  auto parent_quad_in     = cgns::get_node_value_by_matching<I4,2>(b,"Zone/QUAD_4_interior/ParentElements");
  auto parent_pos_quad_in = cgns::get_node_value_by_matching<I4,2>(b,"Zone/QUAD_4_interior/ParentElementsPosition");

  CHECK( elt_type_quad_in == (I4)cgns::QUAD_4 );
  CHECK( range_quad_in == std::vector<I4>{16,18} );
  ELOG(connec_quad_in);
  ELOG(parent_quad_in);
  ELOG(parent_pos_quad_in);
  //MPI_CHECK(0, connec_quad_in == std::vector<I4>{2,7,10,5, 6,7,10,9, 7,12,15,10} ); // Note: the first and last faces are flipped
  //                                                                                  // compared to sequential...
  //MPI_CHECK(1, connec_quad_in == std::vector<I4>{} );
  //MPI_CHECK(0, parent_quad_in.extent() == std_e::multi_index<I8,2>{3,2} );
  //MPI_CHECK(1, parent_quad_in.extent() == std_e::multi_index<I8,2>{0,2} );
  //MPI_CHECK(0, parent_quad_in == cgns::md_array<I4,2>{{17,15},{15,16},{18,16}} ); // ... and so are the parent elements. So this is coherent
  //                                                                                // The difference comes from the fact that we use std::sort,
  //                                                                                // not std::stable_sort
  //MPI_CHECK(1, parent_quad_in == cgns::md_array<I4,2>(0,2) );
  //MPI_CHECK(0, parent_pos_quad_in == cgns::md_array<I4,2>{{ 3, 3},{ 6, 1},{ 3, 3}} );
  //MPI_CHECK(1, parent_pos_quad_in == cgns::md_array<I4,2>(0,2) );
  MPI_CHECK(0, connec_quad_in == std::vector<I4>{2,7,10,5} ); // Note: the first and last faces are flipped
                                                              // compared to sequential...
  MPI_CHECK(1, connec_quad_in == std::vector<I4>{6,7,10,9, 7,12,15,10} );
  MPI_CHECK(0, parent_quad_in.extent() == std_e::multi_index<I8,2>{1,2} );
  MPI_CHECK(1, parent_quad_in.extent() == std_e::multi_index<I8,2>{2,2} );
  MPI_CHECK(0, parent_quad_in == cgns::md_array<I4,2>{{17,15}} ); // ... and so are the parent elements. So this is coherent
                                                                 // The difference comes from the fact that we use std::sort,
                                                                 // not std::stable_sort
  MPI_CHECK(1, parent_quad_in == cgns::md_array<I4,2>{{15,16},{18,16}} );
  MPI_CHECK(0, parent_pos_quad_in == cgns::md_array<I4,2>{{ 3, 3}} );
  MPI_CHECK(1, parent_pos_quad_in == cgns::md_array<I4,2>{{ 6, 1},{ 3, 3}} );

  // hex
  auto hex_cell_face = cgns::get_node_value_by_matching<I4>(b,"Zone/Hexas/CellFace");
  MPI_CHECK(0, hex_cell_face == std::vector<I4>{11,5,16,9,1,17                 } );
  MPI_CHECK(1, hex_cell_face == std::vector<I4>{                17,7,18,10,2,12} );
  // prisms
  auto prism_cell_face = cgns::get_node_value_by_matching<I4>(b,"Zone/Prisms/CellFace");
  MPI_CHECK(0, prism_cell_face == std::vector<I4>{6,3,16,13,15              } );
  MPI_CHECK(1, prism_cell_face == std::vector<I4>{              8,4,18,15,14} );
}
