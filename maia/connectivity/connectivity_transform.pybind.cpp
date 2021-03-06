#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename g_num>
void pe_cgns_to_pdm_face_cell(py::array_t<g_num, py::array::f_style>& pe,
                              py::array_t<g_num, py::array::f_style>& face_cell){
  assert(pe.ndim()        == 2        );
  assert(face_cell.ndim() == 1        );
  assert(face_cell.size() == pe.size());

  int n_face = pe.shape()[0];

  auto pe_ptr        = pe       .template mutable_unchecked<2>();
  auto face_cell_ptr = face_cell.template mutable_unchecked<1>();

  for(int i_face = 0; i_face < n_face; ++i_face){
    face_cell_ptr(2*i_face  ) = pe_ptr(i_face,0);
    face_cell_ptr(2*i_face+1) = pe_ptr(i_face,1);
  }
}

template<typename g_num>
void pdm_face_cell_to_pe_cgns(py::array_t<g_num, py::array::f_style>& face_cell,
                              py::array_t<g_num, py::array::f_style>& pe){
  assert(pe.ndim()        == 2        );
  assert(face_cell.ndim() == 1        );
  assert(face_cell.size() == pe.size());

  int n_face = pe.shape()[0];

  auto pe_ptr        = pe       .template mutable_unchecked<2>();
  auto face_cell_ptr = face_cell.template mutable_unchecked<1>();

  for(int i_face = 0; i_face < n_face; ++i_face){
    pe_ptr(i_face,0) = face_cell_ptr(2*i_face  );
    pe_ptr(i_face,1) = face_cell_ptr(2*i_face+1);
  }
}

template<typename g_num>
void compute_idx_local(py::array_t<int32_t, py::array::f_style>& connect_l_idx,
                       py::array_t<g_num  , py::array::f_style>& connect_g_idx,
                       py::array_t<g_num  , py::array::f_style>& distrib){
  assert(distrib.size() == 2 || distrib.size() == 3);  /* beg_rank / end_rank / n_tot */

  int n_connect = connect_l_idx.shape()[0]-1; /* shape = n_connect + 1 */

  auto distrib_ptr       = distrib      .template mutable_unchecked<1>();
  auto connect_l_idx_ptr = connect_l_idx.template mutable_unchecked<1>();
  auto connect_g_idx_ptr = connect_g_idx.template mutable_unchecked<1>();
  // auto distrib_ptr = connect_g_idx.template mutable_unchecked<1>();

  auto beg_connect = distrib_ptr[0];
  auto dn_connect  = distrib_ptr[1] - beg_connect;

  for(int i = 0; i < n_connect; ++i){
    connect_l_idx_ptr(i) = connect_g_idx_ptr(i) - beg_connect;
  }
  assert(connect_l_idx_ptr[0] == 0);
  connect_l_idx_ptr[n_connect] = dn_connect;

}

template<typename T> auto
make_raw_view(py::array_t<T, py::array::f_style>& x){
  py::buffer_info buf = x.request();
  return static_cast<T*>(buf.ptr);
}

template<typename fld_type>
std::tuple<py::array_t<fld_type, py::array::f_style>, py::array_t<fld_type, py::array::f_style>, py::array_t<fld_type, py::array::f_style>>
interlaced_to_tuple_coords(py::array_t<fld_type, py::array::f_style>& np_xyz){

  int size = np_xyz.size()/3;
  py::array_t<fld_type, py::array::f_style> np_coord_x(size);
  py::array_t<fld_type, py::array::f_style> np_coord_y(size);
  py::array_t<fld_type, py::array::f_style> np_coord_z(size);

  auto coord_xyz = make_raw_view(np_xyz);
  auto coord_x   = make_raw_view(np_coord_x);
  auto coord_y   = make_raw_view(np_coord_y);
  auto coord_z   = make_raw_view(np_coord_z);

  for(int i = 0; i < size; ++i) {
    int offset = 3*i;
    coord_x[i] = coord_xyz[offset  ];
    coord_y[i] = coord_xyz[offset+1];
    coord_z[i] = coord_xyz[offset+2];
  }

  return std::make_tuple(np_coord_x, np_coord_y, np_coord_z);
}


PYBIND11_MODULE(connectivity_transform, m) {
  m.doc() = "pybind11 connectivity_transform plugin"; // optional module docstring

  m.def("pe_cgns_to_pdm_face_cell", &pe_cgns_to_pdm_face_cell<int32_t>,
        py::arg("pe"       ).noconvert(),
        py::arg("face_cell").noconvert());
  m.def("pe_cgns_to_pdm_face_cell", &pe_cgns_to_pdm_face_cell<int64_t>,
        py::arg("pe"       ).noconvert(),
        py::arg("face_cell").noconvert());

  m.def("pdm_face_cell_to_pe_cgns", &pdm_face_cell_to_pe_cgns<int32_t>,
        py::arg("face_cell").noconvert(),
        py::arg("pe"       ).noconvert());
  m.def("pdm_face_cell_to_pe_cgns", &pdm_face_cell_to_pe_cgns<int64_t>,
        py::arg("face_cell").noconvert(),
        py::arg("pe"       ).noconvert());

  m.def("compute_idx_local", &compute_idx_local<int32_t>,
        py::arg("connect_l_idx").noconvert(),
        py::arg("connect_g_idx").noconvert(),
        py::arg("distrib"      ).noconvert());
  m.def("compute_idx_local", &compute_idx_local<int64_t>,
        py::arg("connect_l_idx").noconvert(),
        py::arg("connect_g_idx").noconvert(),
        py::arg("distrib"      ).noconvert());

  m.def("interlaced_to_tuple_coords", &interlaced_to_tuple_coords<double>,
        py::arg("np_xyz").noconvert());

}
