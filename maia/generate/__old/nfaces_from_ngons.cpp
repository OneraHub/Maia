#if __cplusplus > 201703L
#include "maia/generate/__old/nfaces_from_ngons.hpp"

#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids/utils.hpp"
#include "std_e/parallel/algorithm/graph/edge_to_adj.hpp"
#include "std_e/parallel/mpi/collective/bcast.hpp"


using cgns::tree;
using cgns::I4;
using cgns::I8;


namespace maia {


template<class I> auto
nfaces_from_ngons(const tree& ngons, MPI_Comm comm) -> tree {
  STD_E_ASSERT(label(ngons)=="Elements_t");
  STD_E_ASSERT(element_type(ngons)==cgns::NGON_n);

  auto first_ngon_id = ElementRange<I>(ngons)[0];
  auto last_ngon_id  = ElementRange<I>(ngons)[1];
  auto parent_elts_md_array = ParentElements<I>(ngons);
  auto elt_distri = ElementDistribution<I>(ngons);
  auto distri_start = elt_distri[0];
  auto parent_elts = view_as_multi_range2(column(parent_elts_md_array,0),column(parent_elts_md_array,1));

  auto left_face  = [first_ngon_id,distri_start](auto&& , I i){ return   first_ngon_id+distri_start+i; };
  auto right_face = [first_ngon_id,distri_start](auto&& , I i){ return -(first_ngon_id+distri_start+i); };
  auto is_real_cell_id = [](I pe){ return pe!=0; };

  auto cells = edge_list_to_adjacency_list(parent_elts,comm,left_face,right_face,is_real_cell_id);
  auto eso = cells.retrieve_offsets();
  auto connec = cells.retrieve_values();

  I first_nface_id = last_ngon_id+1;
  I dn_cell = eso.size()-1; // -1 because eso has one more element
  I n_cell = std_e::all_reduce(dn_cell,MPI_SUM,comm);
  I start_cell = std_e::ex_scan(dn_cell,MPI_SUM,0,comm);
  I n_connec = std_e::bcast(eso.back(),std_e::n_rank(comm)-1,comm);
  std::vector<I> partial_distri = {start_cell,start_cell+dn_cell,n_cell};
  std::vector<I> partial_distri_connec = {eso[0],eso.back(),n_connec};
  I last_nface_id  = first_nface_id+n_cell -1; // -1 because CGNS ranges are [close,close]
  auto nface_node = cgns::new_Elements(to_string(cgns::NFACE_n),cgns::NFACE_n,std::move(connec),first_nface_id,last_nface_id);
  tree eso_node = cgns::new_DataArray("ElementStartOffset",std::move(eso));
  emplace_child(nface_node,std::move(eso_node));

  auto distri_node = cgns::new_ElementDistribution(std::move(partial_distri),std::move(partial_distri_connec));
  emplace_child(nface_node,std::move(distri_node));

  return nface_node;
}

template<class I> auto
_add_nfaces(tree& z, MPI_Comm comm) -> void {
  emplace_child(z,nfaces_from_ngons<I>(element_section(z,cgns::NGON_n),comm));
}

auto
add_nfaces(tree& z, MPI_Comm comm) -> void {
  return _add_nfaces<I4>(z,comm);
  //if (value(z).data_type()=="I4") return _add_nfaces<I4>(z,comm);
  //if (value(z).data_type()=="I8") return _add_nfaces<I8>(z,comm);
  //throw cgns::cgns_exception("Zone "+name(z)+" has a value of data type "+value(z).data_type()+" but it should be I4 or I8");
}


} // maia
#endif // C++>17
