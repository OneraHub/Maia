#include "maia/transform/__old/partition_with_boundary_first/boundary_vertices_at_beginning.hpp"

#include "maia/utils/log/log.hpp"
#include "maia/connectivity/iter_cgns/range.hpp"
#include "std_e/algorithm/id_permutations.hpp"
#include "range/v3/view/indices.hpp"
#include "range/v3/range/conversion.hpp"
#include "cpp_cgns/node_manip.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"
#include "std_e/utils/switch.hpp"
#include "cpp_cgns/sids/connectivity_category.hpp"
#include "std_e/base/lift.hpp"


namespace cgns {


template<class I> auto
vertex_permutation_to_move_boundary_at_beginning(I nb_of_vertices, const std::vector<I>& boundary_vertex_ids) -> std::vector<I> {
  std::vector<bool> vertices_are_on_boundary(nb_of_vertices,false);
  for (auto boundary_vertex_id : boundary_vertex_ids) {
    I boundary_vertex_index = boundary_vertex_id - 1; // C++ is 0-indexed, CGNS ids are 1-indexed
    vertices_are_on_boundary[boundary_vertex_index] = true;
  }

  auto vertex_permutation = ranges::views::indices(nb_of_vertices) | ranges::to<std::vector<I>>; // init with no permutation
  std::partition(vertex_permutation.begin(),vertex_permutation.end(),[&](auto i){ return vertices_are_on_boundary[i]; });
  return vertex_permutation;
}


template<int connectivity_cat, class I> auto
update_ids_for_elt_type(std::integral_constant<int,connectivity_cat>, tree& elt_pool, const std::vector<I>& vertex_permutation) -> void {
  constexpr auto cat = static_cast<connectivity_category>(connectivity_cat);
  auto vertex_range = connectivity_vertex_range<I,cat>(elt_pool);

  auto perm_old_to_new = std_e::inverse_permutation(vertex_permutation);
  I offset = 1; // CGNS ids begin at 1
  std_e::offset_permutation perm(offset,perm_old_to_new);
  std_e::apply(perm,vertex_range);
}

template<class I> auto
re_number_vertex_ids_in_elements(tree& elt_pool, const std::vector<I>& vertex_permutation) -> void {
  // Preconditions
  //   - vertex_permutation is an index permutation (i.e. sort(permutation) == std_e::iota(permutation.size()))
  //   - any vertex "v" of "elt_pool" is referenced in "vertex_permutation",
  //     i.e. vertex_permutation[v-1] is valid ("-1" because of 1-indexing)
  auto _ = maia_time_log("re_number_vertex_ids_in_elements");

  auto elt_cat = connectivity_category_of<I>(elt_pool);

  // we want to call different instanciations of update_ids_for_elt_type based on elt_cat
  // instanciations are compile-time, so we can't use a run-time select
  // we could use the switch keyword, but the different cases would have the same syntax, only the instanciated types would differ
  // std_e::switch_ is here to do the same but with no duplication
  std_e::switch_<all_connectivity_categories>(elt_cat).apply( LIFT(update_ids_for_elt_type), elt_pool,vertex_permutation );
}


// explicit instanciations (do not pollute the header for only 2 instanciations)
template auto vertex_permutation_to_move_boundary_at_beginning(I4 nb_of_vertices, const std::vector<I4>& boundary_vertex_ids) -> std::vector<I4>;
template auto vertex_permutation_to_move_boundary_at_beginning(I8 nb_of_vertices, const std::vector<I8>& boundary_vertex_ids) -> std::vector<I8>;
template auto re_number_vertex_ids_in_elements(tree& elt_pool, const std::vector<I4>& vertex_permutation) -> void;
template auto re_number_vertex_ids_in_elements(tree& elt_pool, const std::vector<I8>& vertex_permutation) -> void;

} // cgns
