#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "cpp_cgns/sids/elements_utils.hpp"
#include "maia/utils/mpi4py.hpp"
#include "pdm_block_to_part.h"
#include "pdm_distrib.h"

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


template<typename g_num>
py::list
redistribute_mixed_to_separated(py::object mpi4py_obj,
                                py::array_t<g_num  , py::array::f_style>& np_count_by_type,
                                py::array_t<g_num  , py::array::f_style>& np_count_by_type_idx,
                                py::array_t<g_num  , py::array::f_style>& np_count_elmt_vtx_n_by_type,
                                py::array_t<g_num  , py::array::f_style>& np_count_elmt_vtx_n_by_type_idx,
                                py::array_t<g_num  , py::array::f_style>& np_g_count_by_type,
                                py::array_t<g_num  , py::array::f_style>& np_new_to_old,
                                py::array_t<int32_t, py::array::f_style>& np_elmt_connectivity_n,
                                py::array_t<g_num  , py::array::f_style>& np_elmt_connectivity)
{
  auto comm = maia::mpi4py_comm_to_comm(mpi4py_obj);

  // We have a unique array that contains by stride the local elements sort by elements
  // We see it as partition
  auto count_by_type                = make_raw_view(np_count_by_type               );
  auto count_by_type_idx            = make_raw_view(np_count_by_type_idx           );
  auto count_elmt_vtx_n_by_type     = make_raw_view(np_count_elmt_vtx_n_by_type    );
  auto count_elmt_vtx_n_by_type_idx = make_raw_view(np_count_elmt_vtx_n_by_type_idx);
  auto g_count_by_type              = make_raw_view(np_g_count_by_type             );
  auto new_to_old                   = make_raw_view(np_new_to_old                  );
  auto elmt_connectivity_n          = make_raw_view(np_elmt_connectivity_n         );
  auto elmt_connectivity            = make_raw_view(np_elmt_connectivity           );

  int n_elemt_kind = np_count_by_type.size();
  // std::vector<PDM_g_num_t*> part_elemts  (n_elemt_kind);
  // std::vector<int        *> part_elemts_n(n_elemt_kind);
  // std::vector<int>          n_elmt(n_elemt_kind);

  // // Generate partition
  // for(int i = 0; i < n_elemt_kind; ++i) {
  //   int beg          = count_by_type_idx[i];
  //   int beg_elmt_vtx = count_elmt_vtx_n_by_type_idx[i];
  //   part_elemts_n[i] = &elmt_connectivity_n[beg];
  //   part_elemts  [i] = &elmt_connectivity  [beg_elmt_vtx];
  // }

  PDM_MPI_Comm pdm_comm = PDM_MPI_mpi_2_pdm_mpi_comm(&comm);
  int i_rank;
  int n_rank;

  PDM_MPI_Comm_rank(pdm_comm, &i_rank);
  PDM_MPI_Comm_size(pdm_comm, &n_rank);

  int dn_elemt_in = count_by_type_idx[n_elemt_kind];
  PDM_g_num_t* distrib_init = PDM_compute_entity_distribution(pdm_comm, dn_elemt_in);

  PDM_g_num_t **pelemts_g_num = (PDM_g_num_t **) malloc(n_elemt_kind * sizeof(PDM_g_num_t *));
  int          *n_elemts      = (int          *) malloc(n_elemt_kind * sizeof(int          ));
  for(int i_kind = 0; i_kind < n_elemt_kind; ++i_kind) {
    PDM_g_num_t* distrib_elmt = PDM_compute_uniform_entity_distribution(pdm_comm, g_count_by_type[i_kind]);
    n_elemts[i_kind]          = distrib_elmt[i_rank+1] - distrib_elmt[i_rank];

    pelemts_g_num[i_kind] = (PDM_g_num_t *) malloc( n_elemts[i_kind] * sizeof(PDM_g_num_t));

    for(int i = 0; i < n_elemts[i_kind]; ++i) {
      pelemts_g_num[i_kind][i] = distrib_elmt[i_rank] + i + 1;
    }

    free(distrib_elmt);
  }


  //
  // PDM_g_num_t
  PDM_block_to_part_t* btp = PDM_block_to_part_create(distrib_init,
                              (const PDM_g_num_t **)  pelemts_g_num,
                                                      n_elemts,
                                                      n_elemt_kind,
                                                      pdm_comm);

  int         **section_elmt_n = NULL;
  PDM_g_num_t **section_elmt   = NULL;
  PDM_block_to_part_exch2(btp,
                          sizeof(PDM_g_num_t),
                          PDM_STRIDE_VAR,
                          elmt_connectivity_n,
             (void *  )   elmt_connectivity,
             (int  ***)  &section_elmt_n,
             (void ***)  &section_elmt);

  py::list sections_list;
  for(int i_kind = 0; i_kind < n_elemt_kind; ++i_kind) {
    free(pelemts_g_num[i_kind]);

    int n_vtx_tot = 0;
    for(int i = 0; i < n_elemts[i_kind]; ++i) {
      n_vtx_tot += section_elmt_n[i_kind][i];
    }
    // int n_vtx = cgns::number_of_nodes(static_cast<int>(i_type));
    // assert(n_vtx_tot == n_elemts[i_kind] * n_vtx)

    py::capsule capsule(section_elmt[i_kind], free);
    py::array_t<PDM_g_num_t> np_section_elmt_vtx({n_vtx_tot}, section_elmt[i_kind], capsule);

    sections_list.append(np_section_elmt_vtx);

    free(section_elmt_n[i_kind]);
    // free(section_elmt[i_kind]);
  }
  free(pelemts_g_num);

  free(distrib_init);
  free(n_elemts);

  PDM_block_to_part_free(btp);

  return sections_list;
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

  m.def("redistribute_mixed_to_separated", &redistribute_mixed_to_separated<PDM_g_num_t>,
        py::arg("mpi4py_obj"                     ).noconvert(),
        py::arg("np_count_by_type"               ).noconvert(),
        py::arg("np_count_by_type_idx"           ).noconvert(),
        py::arg("np_count_elmt_vtx_n_by_type"    ).noconvert(),
        py::arg("np_count_elmt_vtx_n_by_type_idx").noconvert(),
        py::arg("np_g_count_by_type"             ).noconvert(),
        py::arg("np_new_to_old"                  ).noconvert(),
        py::arg("np_elmt_connectivity_n"         ).noconvert(),
        py::arg("np_elmt_connectivity"           ).noconvert());
  // m.def("redistribute_mixed_to_separated", &redistribute_mixed_to_separated<int64_t>,
  //       py::arg("mpi4py_obj"                     ).noconvert(),
  //       py::arg("np_count_by_type"               ).noconvert(),
  //       py::arg("np_count_by_type_idx"           ).noconvert(),
  //       py::arg("np_count_elmt_vtx_n_by_type"    ).noconvert(),
  //       py::arg("np_count_elmt_vtx_n_by_type_idx").noconvert(),
  //       py::arg("np_new_to_old"                  ).noconvert(),
  //       py::arg("np_elmt_connectivity_n"         ).noconvert(),
  //       py::arg("np_elmt_connectivity"           ).noconvert());
}

