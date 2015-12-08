
#include <iostream>
#include "KokkosKernelsGraphHelpers.hpp"
#include "SPGEMM.hpp"
#include "experiment_space.hpp"


template <typename v1>
struct compare{
  v1 f,s;
  compare (v1 f_ , v1 s_): f(f_), s(s_){}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t &i, size_t &diff) const {

    if (f[i] - s[i] > 0.00001 || f[i] - s[i] < -0.00001) diff++;
  }

};




int main (int argc, char ** argv){
  if (argc < 2){
    std::cerr << "Usage:" << argv[0] << " input_bin_file" << std::endl;
    exit(1);
  }

  Kokkos::initialize(argc, argv);
  MyExecSpace::print_configuration(std::cout);

  idx m = 0, nnzA = 0, n = 0, k = 0;
  idx *xadj, *adj;
  wt *ew;
  KokkosKernels::Experimental::Graph::Utils::read_graph_bin<idx, wt> (&m, &nnzA, &xadj, &adj, &ew, argv[1]);

  std::cout << "m:" << m << " nnzA:" << nnzA << std::endl;
  k = n = m;


  um_array_type _xadj (xadj, m + 1);
  um_edge_array_type _adj (adj, nnzA);

  wt_um_edge_array_type _mtx_vals (ew, nnzA);


  idx_array_type kok_xadj ("xadj", m + 1);
  idx_edge_array_type kok_adj("adj", nnzA);
  value_array_type kok_mtx_vals ("MTX_VALS", nnzA);

  Kokkos::deep_copy (kok_xadj, _xadj);
  Kokkos::deep_copy (kok_adj, _adj);
  Kokkos::deep_copy (kok_mtx_vals, _mtx_vals);


  delete [] xadj;
  delete [] adj;
  delete [] ew;

  idx_array_type row_mapC, row_mapC2;
  idx_edge_array_type entriesC, entriesC2;
  value_array_type valuesC, valuesC2;

  typedef KokkosKernels::Experimental::KokkosKernelsHandle
        <idx_array_type,idx_edge_array_type, value_array_type,
        MyExecSpace, TemporaryWorkSpace,PersistentWorkSpace > KernelHandle;

  KernelHandle kh;



  kh.create_spgemm_handle(KokkosKernels::Experimental::Graph::SPGEMM_CUSP);
  Kokkos::Impl::Timer timer1;
  KokkosKernels::Experimental::Graph::spgemm_symbolic<KernelHandle> (
      &kh,
      m,
      n,
      k,
      kok_xadj,
      kok_adj,
      false,
      kok_xadj,
      kok_adj,
      false,
      row_mapC,
      entriesC
      );

  Kokkos::fence();
  double symbolic_time = timer1.seconds();
  Kokkos::Impl::Timer timer2;
  KokkosKernels::Experimental::Graph::spgemm_numeric(
      &kh,
      m,
      n,
      k,
      kok_xadj,
      kok_adj,
      kok_mtx_vals,
      false,

      kok_xadj,
      kok_adj,
      kok_mtx_vals,
      true,
      row_mapC,
      entriesC,
      valuesC
      );
  Kokkos::fence();
  double numeric_time = timer2.seconds();
  std::cout << "mm_time:" << numeric_time + symbolic_time
            << " symbolic_time:" << symbolic_time
            << " numeric:" << numeric_time << std::endl;

  std::cout << "row_mapC:" << row_mapC.dimension_0() << std::endl;
  std::cout << "entriesC:" << entriesC.dimension_0() << std::endl;
  std::cout << "valuesC:" << valuesC.dimension_0() << std::endl;




  kh.create_spgemm_handle();
  Kokkos::Impl::Timer timer3;
  KokkosKernels::Experimental::Graph::spgemm_symbolic<KernelHandle> (
      &kh,
      m,
      n,
      k,
      kok_xadj,
      kok_adj,
      false,
      kok_xadj,
      kok_adj,
      false,
      row_mapC2,
      entriesC2
      );

  Kokkos::fence();
  symbolic_time = timer3.seconds();
  Kokkos::Impl::Timer timer4;
  KokkosKernels::Experimental::Graph::spgemm_numeric(
      &kh,
      m,
      n,
      k,
      kok_xadj,
      kok_adj,
      kok_mtx_vals,
      false,

      kok_xadj,
      kok_adj,
      kok_mtx_vals,
      true,
      row_mapC2,
      entriesC2,
      valuesC2
      );
  Kokkos::fence();
  numeric_time = timer4.seconds();
  std::cout << "mm_time:" << numeric_time + symbolic_time
            << " symbolic_time:" << symbolic_time
            << " numeric:" << numeric_time << std::endl;

  std::cout << "row_mapC:" << row_mapC2.dimension_0() << std::endl;
  std::cout << "entriesC:" << entriesC2.dimension_0() << std::endl;
  std::cout << "valuesC:" << valuesC2.dimension_0() << std::endl;

  typedef Kokkos::RangePolicy<MyExecSpace> my_exec_space;

  size_t map = 0, ent = 0, val = 0;
  Kokkos::parallel_reduce(my_exec_space(0,row_mapC2.dimension_0()), compare<idx_array_type>(row_mapC,row_mapC2), map);
  Kokkos::parallel_reduce(my_exec_space(0,entriesC2.dimension_0()), compare<idx_edge_array_type>(entriesC,entriesC2), ent);
  Kokkos::parallel_reduce(my_exec_space(0,valuesC2.dimension_0()), compare<value_array_type>(valuesC,valuesC2), val);

  std::cout << "map:" << map << " ent:" << ent << " val:" << val << std::endl;
  Kokkos::finalize();
  return 0;

}


