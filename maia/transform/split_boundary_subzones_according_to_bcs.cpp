#include "maia/transform/split_boundary_subzones_according_to_bcs.hpp"
#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids/sids.hpp"
#include <algorithm>
#include <vector>
#include "maia/partitioning/gc_name_convention.hpp"
#include "std_e/buffer/buffer_vector.hpp"
#include "std_e/utils/concatenate.hpp"
#include "maia/utils/parallel/utils.hpp"

#include "maia/utils/parallel/exchange/block_to_part.hpp"
#include "std_e/log.hpp" // TODO


using namespace cgns;


namespace maia {


// TODO I4 -> I, R8 -> R
template<class Tree_range, class Distribution> auto
sub_field_for_ids(const Tree_range& fields, std_e::span<const I4> ids, I4 first_id, const Distribution& distri, MPI_Comm comm) {
  MPI_Barrier(comm);
  LOG("\n\n\n");
  MPI_Barrier(comm);
  I4 sz = ids.size();
  //ELOG(sz);

  ELOG(distri)
  ELOG(ids.size())
  pdm::block_to_part_protocol btp_protocol(comm,distri,ids);
  std::vector<tree> sub_field_nodes;
  for (const tree& field_node : fields) {
    MPI_Barrier(comm);
    LOG("\n\n\n");
    ELOG(field_node)
    MPI_Barrier(comm);
    ELOG(field_node.name);
    auto dfield = cgns::view_as_span<R8>(field_node.value);
    ELOG(dfield.size());
    auto pfield = pdm::exchange(btp_protocol,dfield);
    ELOG(pfield.size());
    std_e::buffer_vector<R8> sub_field(sz);
    std::copy(begin(pfield),end(pfield),begin(sub_field));
    //if (field_node.name == "SurfaceNormalX") {
    //    ELOG(dfield.size())
    //    ELOG(sub_field.size())
    //    ELOG(sub_field)
    //}

    sub_field_nodes.push_back(cgns::new_DataArray(field_node.name,std::move(sub_field)));
  }
  LOG("tata");

  return sub_field_nodes;
}

template<class Tree_range> auto
split_boundary_subzone_according_to_bcs(const tree& zsr, const Tree_range& bcs, MPI_Comm comm) {
  I4 first_id = get_child_value_by_name<I4,2>(zsr,"PointRange")(0,0);
  auto partial_dist = get_node_value_by_matching<I8>(zsr,":CGNS#Distribution/Index");
  ELOG(partial_dist);
  auto distri = distribution_from_partial(partial_dist,comm);
  std::vector<tree> sub_zsrs;
  for (const tree& bc : bcs) {
    ELOG(bc.name);
    auto& pl = get_child_by_name(bc,"PointList");
    auto fields_on_bc = sub_field_for_ids(get_children_by_label(zsr,"DataArray_t"),view_as_span<I4>(pl.value),first_id,distri,comm);
      LOG("tata01");

    auto sub_zsr = new_ZoneSubRegion(zsr.name+"_"+bc.name,2,"FaceCenter");
    ELOG(sub_zsr.name);
    emplace_children(sub_zsr,std::move(fields_on_bc));
    emplace_child(sub_zsr,new_Descriptor("BCRegionName",bc.name));

    sub_zsrs.push_back(std::move(sub_zsr));
  }
  return sub_zsrs;
}



auto
is_elt_section_2D(const tree& n) -> bool {
  return label(n)=="Elements_t" && element_dimension(element_type(n))==2;
}

auto
boundary_elements_interval(const tree& z) {
  auto bnd_elts = get_children_by_predicate(z,is_elt_section_2D);
  std::sort(begin(bnd_elts),end(bnd_elts),cgns::compare_by_range);

  if (!cgns::elts_ranges_are_contiguous(bnd_elts)) {
    throw cgns_exception("2D Elements_t ranges are expected to be contiguous");
  }

  return
    cgns::interval<I8>(
      element_range(bnd_elts[0]).first(),
      element_range(bnd_elts.back()).last()
    );
}

auto
is_complete_bnd_zone_sub_region(const tree& t, const cgns::interval<I8>& elt_2d_range) -> bool {
  if (label(t)!="ZoneSubRegion_t") return false;
  if (GridLocation(t)!="FaceCenter") return false;
  if (!has_child_of_name(t,"PointRange")) return false;
  auto range = point_range_to_interval(get_child_by_name(t,"PointRange"));
  if (range==elt_2d_range) return true;
  return false;
}

auto
is_bc_on_faces(const tree& t) -> bool {
  if (label(t)!="BC_t") return false;
  return GridLocation(t)=="FaceCenter";
}

auto
split_boundary_subzones_according_to_bcs(tree& b, MPI_Comm comm) -> void {
  auto zs = get_children_by_label(b,"Zone_t");
  for (tree& z : zs) {
    auto elt_2d_range = boundary_elements_interval(z);
    ELOG(z.name);
    auto zsrs = get_children_by_predicate(z,[&](const tree& t){ return is_complete_bnd_zone_sub_region(t,elt_2d_range); });

    auto& zbc = get_child_by_name(z,"ZoneBC");
    auto bcs = get_children_by_predicate(zbc,is_bc_on_faces);

    std::vector<tree> all_sub_zsrs;
    for (const tree& zsr : zsrs) {
      ELOG(zsr.name);
      auto sub_zsrs = split_boundary_subzone_according_to_bcs(zsr,bcs,comm);
      std_e::emplace_back(all_sub_zsrs,std::move(sub_zsrs));
    }

    // change children at the end (else, iterator invalidation)
    rm_children(z,zsrs);
    emplace_children(z,std::move(all_sub_zsrs));
  }
}


} // maia
