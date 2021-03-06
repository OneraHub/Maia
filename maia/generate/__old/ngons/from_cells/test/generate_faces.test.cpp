#include "std_e/unit_test/doctest.hpp"

#include "maia/generate/__old/ngons/from_cells/generate_faces.hpp"
#include "maia/connectivity/iter_cgns/connectivity.hpp"

using namespace cgns;

TEST_CASE("generate_faces: face") {
  quad_4<int> quad = {3,9,12,1};
  
  auto faces = generate_faces(quad);

  CHECK(std::get<0>(faces) == quad_4<int>{3,9,12,1} );
}

TEST_CASE("generate_faces: tet") {
  tet_4<int> tet = {3,9,12,1};

  auto faces = generate_faces(tet);

  CHECK( std::get<0>(faces) == tri_3<int>{ 3,12, 9} );
  CHECK( std::get<1>(faces) == tri_3<int>{ 3, 9, 1} );
  CHECK( std::get<2>(faces) == tri_3<int>{ 3, 1,12} );
  CHECK( std::get<3>(faces) == tri_3<int>{ 9,12, 1} );
}

TEST_CASE("generate_faces: pyra") {
  pyra_5<int> pyra = {3,9,12,1,4};

  auto faces = generate_faces(pyra);

  CHECK( std::get<0>(faces) == quad_4<int>{ 3, 1,12, 9} );
  CHECK( std::get<1>(faces) == tri_3<int>{ 3, 9, 4} );
  CHECK( std::get<2>(faces) == tri_3<int>{ 9,12, 4} );
  CHECK( std::get<3>(faces) == tri_3<int>{12, 1, 4} );
  CHECK( std::get<4>(faces) == tri_3<int>{ 1, 3, 4} );
}

TEST_CASE("generate_faces: prism") {
  penta_6<int> prism = {12,9,3,2,4,6};

  auto faces = generate_faces(prism);

  CHECK( std::get<0>(faces) == tri_3<int>{12, 3, 9} );
  CHECK( std::get<1>(faces) == tri_3<int>{ 2, 4, 6} );

  CHECK( std::get<2>(faces) == quad_4<int>{12, 9, 4, 2} );
  CHECK( std::get<3>(faces) == quad_4<int>{12, 2, 6, 3} );
  CHECK( std::get<4>(faces) == quad_4<int>{ 9, 3, 6, 4} );
}

TEST_CASE("generate_faces: homogenous_hex_connectivities") {
  hex_8<int> hex = {3,9,12,1,4,8,15,17};

  auto faces = generate_faces(hex);
  
  CHECK( std::get<0>(faces) == quad_4<int>{ 3, 1,12, 9} );
  CHECK( std::get<1>(faces) == quad_4<int>{ 4, 8,15,17} );
  CHECK( std::get<2>(faces) == quad_4<int>{ 3, 9, 8, 4} );
  CHECK( std::get<3>(faces) == quad_4<int>{ 3, 4,17, 1} );
  CHECK( std::get<4>(faces) == quad_4<int>{ 9,12,15, 8} );
  CHECK( std::get<5>(faces) == quad_4<int>{12, 1,17,15} );
}
