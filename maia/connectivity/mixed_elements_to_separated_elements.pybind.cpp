#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "cpp_cgns/sids/elements_utils.hpp"

namespace py = pybind11;

template<typename T> auto
make_raw_view(py::array_t<T, py::array::f_style>& x){
  py::buffer_info buf = x.request();
  return static_cast<T*>(buf.ptr);
}

template<typename g_num>
int
count_type_in_mixed_elements(py::array_t<g_num  , py::array::f_style>& np_count_by_type,
                             py::array_t<g_num  , py::array::f_style>& np_count_elmt_vtx_n_by_type,
                             py::array_t<g_num  , py::array::f_style>& np_elmt_range,
                             py::array_t<g_num  , py::array::f_style>& np_elmt_connectivity_idx,
                             py::array_t<g_num  , py::array::f_style>& np_elmt_connectivity,
                             py::array_t<int64_t, py::array::f_style>& np_distrib_elmt)
{
  assert(np_distrib_elmt.size() == 3); // start, dn, n_g

  auto count_by_type            = make_raw_view(np_count_by_type        );
  auto count_elmt_vtx_n_by_type = make_raw_view(np_count_elmt_vtx_n_by_type        );
  auto elmt_range               = make_raw_view(np_elmt_range           );
  auto elmt_connectivity_idx    = make_raw_view(np_elmt_connectivity_idx);
  auto elmt_connectivity        = make_raw_view(np_elmt_connectivity    );
  auto distrib_elmt             = make_raw_view(np_distrib_elmt         );

  int64_t beg_elmt = distrib_elmt[0];
  int64_t dn_elmt  = distrib_elmt[1];

  assert(dn_elmt == static_cast<int64_t>(np_elmt_connectivity_idx.size()-1));

  int n_elmt_vtx_tot = 0;
  for(int i_elemt = 0; i_elemt < dn_elmt; ++i_elemt) {
    int beg    = elmt_connectivity_idx[i_elemt] - beg_elmt;
    int i_type = elmt_connectivity[beg];
    count_by_type[i_type]++;

    int n_vtx = cgns::number_of_nodes(static_cast<int>(i_type));
    count_elmt_vtx_n_by_type[i_type] += n_vtx;
    n_elmt_vtx_tot                   += n_vtx;

  }

  return n_elmt_vtx_tot;
}


template<typename g_num>
void sort_by_type(py::array_t<g_num  , py::array::f_style>& np_count_by_type,
                  py::array_t<g_num  , py::array::f_style>& np_count_by_type_idx,
                  py::array_t<g_num  , py::array::f_style>& np_count_elmt_vtx_n_by_type,
                  py::array_t<g_num  , py::array::f_style>& np_count_elmt_vtx_n_by_type_idx,
                  py::array_t<g_num  , py::array::f_style>& np_elmt_range,
                  py::array_t<g_num  , py::array::f_style>& np_elmt_connectivity_idx,
                  py::array_t<g_num  , py::array::f_style>& np_elmt_connectivity,
                  py::array_t<int64_t, py::array::f_style>& np_distrib_elmt,
                  py::array_t<g_num  , py::array::f_style>& np_new_to_old,
                  py::array_t<int32_t, py::array::f_style>& np_elmt_connectivity_n_out,
                  py::array_t<g_num  , py::array::f_style>& np_elmt_connectivity_out){

  assert(np_distrib_elmt.size() == 3); // start, dn, n_g

  auto count_by_type                = make_raw_view(np_count_by_type               );
  auto count_by_type_idx            = make_raw_view(np_count_by_type_idx           );
  auto count_elmt_vtx_n_by_type     = make_raw_view(np_count_elmt_vtx_n_by_type    );
  auto count_elmt_vtx_n_by_type_idx = make_raw_view(np_count_elmt_vtx_n_by_type_idx);
  auto elmt_range                   = make_raw_view(np_elmt_range                  );
  auto elmt_connectivity_idx        = make_raw_view(np_elmt_connectivity_idx       );
  auto elmt_connectivity            = make_raw_view(np_elmt_connectivity           );
  auto distrib_elmt                 = make_raw_view(np_distrib_elmt                );
  auto new_to_old                   = make_raw_view(np_new_to_old                  );
  auto elmt_connectivity_n_out      = make_raw_view(np_elmt_connectivity_n_out     );
  auto elmt_connectivity_out        = make_raw_view(np_elmt_connectivity_out       );

  int64_t beg_elmt = distrib_elmt[0];
  int64_t dn_elmt  = distrib_elmt[1];

  assert(dn_elmt == static_cast<int64_t>(np_elmt_connectivity_idx.size()-1));

  for(int i_elemt = 0; i_elemt < dn_elmt; ++i_elemt) {
    int beg    = elmt_connectivity_idx[i_elemt] - beg_elmt;
    int i_type = elmt_connectivity[beg];

    int n_vtx = cgns::number_of_nodes(static_cast<int>(i_type));
    int beg_write_elmt_connectivity = count_elmt_vtx_n_by_type_idx[i_type] + count_elmt_vtx_n_by_type[i_type];
    for(int j = 0; j < n_vtx; ++j) {
      elmt_connectivity_out[beg_write_elmt_connectivity+j] = elmt_connectivity[beg+j+1]; // +1 stand to skip the type
    }

    int beg_write = count_by_type_idx[i_type] + count_by_type[i_type]++;
    elmt_connectivity_n_out[beg_write] = n_vtx;

    new_to_old[beg_write] = elmt_range[0] + beg_elmt + i_elemt;

    count_elmt_vtx_n_by_type[i_type] += n_vtx;

  }

}



template<typename g_num>
py::array_t<g_num, py::array::f_style>
idx_from_count(py::array_t<g_num  , py::array::f_style>& np_count){

  int size = np_count.size();
  py::array_t<g_num, py::array::f_style> np_idx(size+1);

  auto count  = make_raw_view(np_count);
  auto idx    = make_raw_view(np_idx  );

  idx[0] = 0;
  for(int i = 0; i < size; ++i) {
    idx[i+1] = idx[i] + count[i];
  }

  return np_idx;
}



PYBIND11_MODULE(mixed_elements_to_separated_elements, m) {
  m.doc() = "pybind11 connectivity_transform plugin"; // optional module docstring

  m.def("count_type_in_mixed_elements", &count_type_in_mixed_elements<int32_t>,
        py::arg("np_count_by_type"           ).noconvert(),
        py::arg("np_count_elmt_vtx_n_by_type").noconvert(),
        py::arg("np_elmt_range"              ).noconvert(),
        py::arg("np_elmt_connectivity_idx"   ).noconvert(),
        py::arg("np_elmt_connectivity"       ).noconvert(),
        py::arg("np_distrib_elmt"            ).noconvert());
  m.def("count_type_in_mixed_elements", &count_type_in_mixed_elements<int64_t>,
        py::arg("np_count_by_type"           ).noconvert(),
        py::arg("np_count_elmt_vtx_n_by_type").noconvert(),
        py::arg("np_elmt_range"              ).noconvert(),
        py::arg("np_elmt_connectivity_idx"   ).noconvert(),
        py::arg("np_elmt_connectivity"       ).noconvert(),
        py::arg("np_distrib_elmt"            ).noconvert());

  m.def("sort_by_type", &sort_by_type<int32_t>,
        py::arg("np_count_by_type"               ).noconvert(),
        py::arg("np_count_by_type_idx"           ).noconvert(),
        py::arg("np_count_elmt_vtx_n_by_type"    ).noconvert(),
        py::arg("np_count_elmt_vtx_n_by_type_idx").noconvert(),
        py::arg("np_elmt_range"                  ).noconvert(),
        py::arg("np_elmt_connectivity_idx"       ).noconvert(),
        py::arg("np_elmt_connectivity"           ).noconvert(),
        py::arg("np_distrib_elmt"                ).noconvert(),
        py::arg("np_new_to_old"                  ).noconvert(),
        py::arg("np_elmt_connectivity_n_out"     ).noconvert(),
        py::arg("np_elmt_connectivity_out"       ).noconvert());
  m.def("sort_by_type", &sort_by_type<int64_t>,
        py::arg("np_count_by_type"               ).noconvert(),
        py::arg("np_count_by_type_idx"           ).noconvert(),
        py::arg("np_count_elmt_vtx_n_by_type"    ).noconvert(),
        py::arg("np_count_elmt_vtx_n_by_type_idx").noconvert(),
        py::arg("np_elmt_range"                  ).noconvert(),
        py::arg("np_elmt_connectivity_idx"       ).noconvert(),
        py::arg("np_elmt_connectivity"           ).noconvert(),
        py::arg("np_distrib_elmt"                ).noconvert(),
        py::arg("np_new_to_old"                  ).noconvert(),
        py::arg("np_elmt_connectivity_n_out"     ).noconvert(),
        py::arg("np_elmt_connectivity_out"       ).noconvert());

  m.def("idx_from_count", &idx_from_count<int32_t>,
        py::arg("np_count").noconvert());
  m.def("idx_from_count", &idx_from_count<int64_t>,
        py::arg("np_count").noconvert());
}

