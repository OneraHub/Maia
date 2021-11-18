#include "maia/transform/fsdm_distribution.hpp"
#include "cpp_cgns/node_manip.hpp"
#include "cpp_cgns/sids/Hierarchical_Structures.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids/sids.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "maia/utils/parallel/distribution.hpp"
#include "maia/utils/parallel/utils.hpp"
#include "maia/sids/element_sections.hpp"
#include "pdm_multi_block_to_part.h"
#include "std_e/algorithm/algorithm.hpp"
#include "std_e/buffer/buffer_vector.hpp"
#include "maia/utils/parallel/exchange/multi_block_to_part.hpp"
#include <algorithm>

using namespace cgns;

namespace maia {

auto add_fsdm_distribution(tree& b, MPI_Comm comm) -> void {
  STD_E_ASSERT(b.label=="CGNSBase_t");
  auto zs = get_children_by_label(b,"Zone_t");
  if (zs.size()!=1) {
    throw cgns_exception("add_fsdm_distribution (as FSDM) expects only one zone per process");
  }
  tree& z = zs[0];

  int n_rank = std_e::n_rank(comm);

  auto n_vtx = VertexSize_U<I4>(z);
  auto n_ghost_node = 0;
  if (cgns::has_node(z,"GridCoordinates/FSDM#n_ghost")) {
    n_ghost_node = get_node_value_by_matching<I4>(z,"GridCoordinates/FSDM#n_ghost")[0];
  }
  I4 n_owned_vtx = n_vtx - n_ghost_node;
  auto vtx_distri = distribution_from_dsizes(n_owned_vtx, comm);
  auto partial_vtx_distri = full_to_partial_distribution(vtx_distri,comm);
  std_e::buffer_vector<I8> vtx_distri_mem(begin(partial_vtx_distri),end(partial_vtx_distri));
  tree vtx_dist = new_DataArray("Vertex",std::move(vtx_distri_mem));
  auto dist_node = new_UserDefinedData(":CGNS#Distribution");
  emplace_child(dist_node,std::move(vtx_dist));
  emplace_child(z,std::move(dist_node));

  auto elt_sections = get_children_by_label(z,"Elements_t");
  for (tree& elt_section : elt_sections) {
    auto elt_range = ElementRange<I4>(elt_section);
    I4 n_owned_elt = elt_range[1] - elt_range[0] + 1;

    auto elt_distri = distribution_from_dsizes(n_owned_elt, comm);
    auto partial_elt_distri = full_to_partial_distribution(elt_distri,comm);
    std_e::buffer_vector<I8> elt_distri_mem(begin(partial_elt_distri),end(partial_elt_distri));

    I4 elt_type = ElementType<I4>(elt_section);
    tree elt_dist = new_DataArray("Element",std::move(elt_distri_mem));

    auto dist_node = new_UserDefinedData(":CGNS#Distribution");
    emplace_child(dist_node,std::move(elt_dist));
    emplace_child(elt_section,std::move(dist_node));
  }
}

template<class I, class Tree_range> auto
elt_interval_range(const Tree_range& sorted_elt_sections) {
  int n_elt = sorted_elt_sections.size();
  std::vector<I> interval_rng(n_elt+1);

  for (int i=0; i<n_elt; ++i) {
    interval_rng[i] = ElementRange<I>(sorted_elt_sections[i])[0];
  }

  interval_rng[n_elt] = ElementRange<I>(sorted_elt_sections.back())[1]+1; // +1 because CGNS intervals are closed, we want open

  return interval_rng;
}

template<class Tree_range> auto
elt_distributions(const Tree_range& sorted_elt_sections, MPI_Comm comm) {
  int n_elt = sorted_elt_sections.size();
  std::vector<distribution_vector<I4>> dists(n_elt);
  for (int i=0; i<n_elt; ++i) {
    const tree& elt = sorted_elt_sections[i];
    auto partial_dist = cgns::get_node_value_by_matching<I8>(elt,":CGNS#Distribution/Element");
    auto dist_I8 = distribution_from_partial(partial_dist,comm);
    dists[i] = distribution_vector<I4>(dist_I8.n_interval()); // TODO make resize accessible
    std::copy(begin(dist_I8),end(dist_I8),begin(dists[i]));
  }
  return dists;
}


auto
distribute_bc_ids_to_match_face_dist(tree& b, MPI_Comm comm) -> void {
  STD_E_ASSERT(b.label=="CGNSBase_t");
  for (tree& z : get_children_by_label(b,"Zone_t")) {
    auto elt_sections = element_sections_ordered_by_range(z);
    auto elt_intervals = elt_interval_range<I4>(elt_sections);
    auto elt_dists = elt_distributions(elt_sections,comm);

    for (tree& bc : cgns::get_nodes_by_matching(z,"ZoneBC/BC_t")) {
      auto pl = cgns::PointList<I4>(bc);
      auto fields = std::vector<std::vector<double>>{}; // TODO extract these fields (if they exist)
      auto [new_pl,_] = redistribute_to_match_face_dist(elt_dists,elt_intervals,pl,fields,comm);

      rm_child_by_name(bc,"PointList");

      std_e::buffer_vector<I4> pl_buf(begin(new_pl),end(new_pl));
      cgns::emplace_child(bc,new_PointList("PointList",std::move(pl_buf)));
    }
    // TODO update BC distribution
  }
}


auto
distribute_vol_fields_to_match_global_element_range(cgns::tree& b, MPI_Comm comm) -> void {
  STD_E_ASSERT(b.label=="CGNSBase_t");
  int i_rank = std_e::rank(comm);
  int n_rank = std_e::n_rank(comm);

  for (tree& z : get_children_by_label(b,"Zone_t")) {
    int n_cell = cgns::CellSize_U<I4>(z);

    auto elt_sections = element_sections_ordered_by_range(z);
    auto elt_intervals = elt_interval_range<I4>(elt_sections);
    auto elt_dists = elt_distributions(elt_sections,comm);

    int n_section = elt_sections.size();

    auto section_is_2d = [](const tree& x){ return element_dimension(element_type(x))==2; };
    STD_E_ASSERT(std::is_partitioned(begin(elt_sections),end(elt_sections),section_is_2d));
    auto first_section_3d = std::partition_point(begin(elt_sections),end(elt_sections),section_is_2d);
    int n_2d_section = first_section_3d - begin(elt_sections);
    auto elt_3d_sections = std_e::make_span(elt_sections.data()+n_2d_section,elt_sections.data()+n_section);

    int n_3d_section = n_section-n_2d_section;

    // multi-block to part
    std::vector<distribution_vector<PDM_g_num_t>> distribs(n_3d_section);
    std::vector<int> d_elt_szs(n_3d_section);
    for (int i=0; i<n_3d_section; ++i) {
      tree& section_node = elt_3d_sections[i];
      auto section_connec_partial_distri = get_node_value_by_matching<I8>(section_node,":CGNS#Distribution/Element");
      d_elt_szs[i] = section_connec_partial_distri[1]-section_connec_partial_distri[0];
      distribs[i] = distribution_from_partial(section_connec_partial_distri,comm);
    }

    const int n_block = n_3d_section;

    std::vector<PDM_g_num_t> merged_distri(n_rank+1);
    std_e::uniform_distribution(begin(merged_distri),end(merged_distri),0,(PDM_g_num_t)n_cell); // TODO uniform_distribution with differing args

    int n_elts = merged_distri[i_rank+1]-merged_distri[i_rank];
    std::vector<PDM_g_num_t> ln_to_gn(n_elts);
    std::iota(begin(ln_to_gn),end(ln_to_gn),merged_distri[i_rank]+1);

    pdm::multi_block_to_part_protocol mbtp(distribs,ln_to_gn,comm);

    tree_range flow_sol_nodes = cgns::get_children_by_labels(z,{"FlowSolution_t","DiscreteData_t"});
    for (tree& flow_sol_node : flow_sol_nodes) {
      tree_range sol_nodes = cgns::get_children_by_label(flow_sol_node,"DataArray_t");
      for (tree& sol_node : sol_nodes) {
        auto sol = cgns::view_as_span<R8>(sol_node.value);
        std::vector<std_e::span<R8>> d_arrays(n_block);
        int offset = 0;
        for (int i=0; i<n_block; ++i) {
          d_arrays[i] = std_e::make_span(sol.data()+offset , d_elt_szs[i]);
          offset += d_elt_szs[i];
        }

        auto p_data = mbtp.exchange(d_arrays);
        std_e::buffer_vector<R8> new_sol(p_data.begin(),p_data.end());

        sol_node.value = cgns::make_node_value(std::move(new_sol));
      }
    }

    tree& z_dist_node = cgns::get_child_by_name(z,":CGNS#Distribution");
    std_e::buffer_vector<I8> cell_partial_dist = {merged_distri[i_rank],merged_distri[i_rank+1],merged_distri.back()};
    emplace_child(z_dist_node,new_DataArray("Cell",std::move(cell_partial_dist)));
  }
}

auto
distribute_fields_to_match_global_element_range(cgns::tree& b, MPI_Comm comm) -> void {
  STD_E_ASSERT(b.label=="CGNSBase_t");
  int i_rank = std_e::rank(comm);
  int n_rank = std_e::n_rank(comm);

  for (tree& z : get_children_by_label(b,"Zone_t")) {
    // 0. preparation
    int n_cell = cgns::CellSize_U<I4>(z);

    auto elt_sections = element_sections_ordered_by_range(z);
    auto elt_intervals = elt_interval_range<I4>(elt_sections);
    auto elt_dists = elt_distributions(elt_sections,comm);

    int n_section = elt_sections.size();

    auto section_is_2d = [](const tree& x){ return element_dimension(element_type(x))==2; };
    STD_E_ASSERT(std::is_partitioned(begin(elt_sections),end(elt_sections),section_is_2d));
    auto first_section_3d = std::partition_point(begin(elt_sections),end(elt_sections),section_is_2d);

    int n_2d_section = first_section_3d - begin(elt_sections);
    int n_3d_section = n_section-n_2d_section;

    auto elt_2d_sections = std_e::make_span(elt_sections.data()             ,elt_sections.data()+n_2d_section);
    auto elt_3d_sections = std_e::make_span(elt_sections.data()+n_2d_section,elt_sections.data()+n_section);

    // 1. multi-block to part for volume fields
    {
      std::vector<distribution_vector<PDM_g_num_t>> distribs(n_3d_section);
      std::vector<int> d_elt_szs(n_3d_section);
      for (int i=0; i<n_3d_section; ++i) {
        tree& section_node = elt_3d_sections[i];
        auto section_connec_partial_distri = get_node_value_by_matching<I8>(section_node,":CGNS#Distribution/Element");
        d_elt_szs[i] = section_connec_partial_distri[1]-section_connec_partial_distri[0];
        distribs[i] = distribution_from_partial(section_connec_partial_distri,comm);
      }

      const int n_block = n_3d_section;

      std::vector<PDM_g_num_t> merged_distri(n_rank+1);
      std_e::uniform_distribution(begin(merged_distri),end(merged_distri),0,(PDM_g_num_t)n_cell); // TODO uniform_distribution with differing args

      int n_elts = merged_distri[i_rank+1]-merged_distri[i_rank];
      std::vector<PDM_g_num_t> ln_to_gn(n_elts);
      std::iota(begin(ln_to_gn),end(ln_to_gn),merged_distri[i_rank]+1);

      pdm::multi_block_to_part_protocol mbtp(distribs,ln_to_gn,comm);

      tree_range flow_sol_nodes = cgns::get_children_by_labels(z,{"FlowSolution_t","DiscreteData_t"});
      for (tree& flow_sol_node : flow_sol_nodes) {
        tree_range sol_nodes = cgns::get_children_by_label(flow_sol_node,"DataArray_t");
        for (tree& sol_node : sol_nodes) {
          auto sol = cgns::view_as_span<R8>(sol_node.value);
          std::vector<std_e::span<R8>> d_arrays(n_block);
          int offset = 0;
          for (int i=0; i<n_block; ++i) {
            d_arrays[i] = std_e::make_span(sol.data()+offset , d_elt_szs[i]);
            offset += d_elt_szs[i];
          }

          auto p_data = mbtp.exchange(d_arrays);
          std_e::buffer_vector<R8> new_sol(p_data.begin(),p_data.end());

          sol_node.value = cgns::make_node_value(std::move(new_sol));
        }
      }

      tree& z_dist_node = cgns::get_child_by_name(z,":CGNS#Distribution");
      std_e::buffer_vector<I8> cell_partial_dist = {merged_distri[i_rank],merged_distri[i_rank+1],merged_distri.back()};
      emplace_child(z_dist_node,new_DataArray("Cell",std::move(cell_partial_dist)));
    }

    // 2. multi-block to part for boundary fields // TODO factor with 1.
    {
      std::vector<distribution_vector<PDM_g_num_t>> distribs(n_2d_section);
      std::vector<int> d_elt_szs(n_2d_section);
      for (int i=0; i<n_2d_section; ++i) {
        tree& section_node = elt_2d_sections[i];
        auto section_connec_partial_distri = get_node_value_by_matching<I8>(section_node,":CGNS#Distribution/Element");
        d_elt_szs[i] = section_connec_partial_distri[1]-section_connec_partial_distri[0];
        distribs[i] = distribution_from_partial(section_connec_partial_distri,comm);
      }

      const int n_block = n_2d_section;

      PDM_g_num_t n_faces = 0;
      if (elt_2d_sections.size() > 0) {
        // since 2d sections are ordered, the total number of 2d elt is the ending of the last range
        n_faces = ElementRange<I4>(elt_2d_sections.back())[1];
      }
      std::vector<PDM_g_num_t> merged_distri(n_rank+1);
      std_e::uniform_distribution(begin(merged_distri),end(merged_distri),0,n_faces); // TODO uniform_distribution with differing args

      int n_elts = merged_distri[i_rank+1]-merged_distri[i_rank];
      std::vector<PDM_g_num_t> ln_to_gn(n_elts);
      std::iota(begin(ln_to_gn),end(ln_to_gn),merged_distri[i_rank]+1);

      pdm::multi_block_to_part_protocol mbtp(distribs,ln_to_gn,comm);

      tree_range bnd_sol_nodes = cgns::get_children_by_label(z,"ZoneSubRegion_t");
      for (tree& bnd_sol_node : bnd_sol_nodes) {
        tree_range sol_nodes = cgns::get_children_by_label(bnd_sol_node,"DataArray_t");
        for (tree& sol_node : sol_nodes) {
          auto sol = cgns::view_as_span<R8>(sol_node.value);
          std::vector<std_e::span<R8>> d_arrays(n_block);
          int offset = 0;
          for (int i=0; i<n_block; ++i) {
            d_arrays[i] = std_e::make_span(sol.data()+offset , d_elt_szs[i]);
            offset += d_elt_szs[i];
          }

          auto p_data = mbtp.exchange(d_arrays);
          std_e::buffer_vector<R8> new_sol(p_data.begin(),p_data.end());

          sol_node.value = cgns::make_node_value(std::move(new_sol));
        }
        std_e::buffer_vector<I8> part_dist = {merged_distri[i_rank],merged_distri[i_rank+1],merged_distri.back()};
        emplace_child(bnd_sol_node,new_Distribution("Index",std::move(part_dist)));
      }
    }
  }
}


} // maia
