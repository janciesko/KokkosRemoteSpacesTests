#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <gtest/gtest.h>
#include <mpi.h>

#include <RDMA_Interface.hpp>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using DeviceSpace_t = Kokkos::CudaSpace;
using HostSpace_t = Kokkos::HostSpace;

using RemoteTraits = Kokkos::RemoteSpaces_MemoryTraitsFlags;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
#ifdef KRS_ENABLE_SHMEMSPACE
  shmem_init_thread(mpi_thread_level_required, &mpi_thread_level_available);
  assert(mpi_thread_level_available >= mpi_thread_level_required);
#endif

  MPI_Comm mpi_comm;

#ifdef KRS_ENABLE_NVSHMEMSPACE
  nvshmemx_init_attr_t attr;
  mpi_comm = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#endif

  Kokkos::initialize(argc, argv);
  {
    int K = argc > 1 ? atoi(argv[1]) : 1024; /*SIZE*/
    int L = argc > 2 ? atoi(argv[2]) : 3;    /*NUM TEAMS*/
    int M = argc > 2 ? atoi(argv[3]) : 3;    /*PARALLEL TEAMS*/

    using Data_t = double;
    int dim0 = K;

    int myRank;
    int numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    MPI_Barrier(MPI_COMM_WORLD);

    using ViewHost_1D_t = Kokkos::View<Data_t *, HostSpace_t>;
    using ViewDevice_1D_t = Kokkos::View<Data_t *, DeviceSpace_t>;
    using ViewRemote_1D_t =
        Kokkos::View<Data_t *, RemoteSpace_t,
                     Kokkos::MemoryTraits<RemoteTraits::Cached>>;

    ViewRemote_1D_t v_r = ViewRemote_1D_t("RemoteView", dim0);
    ViewDevice_1D_t v_d = ViewDevice_1D_t(v_r.data(), v_r.extent(0));
    ViewDevice_1D_t v_d_out_1 = ViewDevice_1D_t("DataView", v_r.extent(0));
    ViewDevice_1D_t v_d_out_2 = ViewDevice_1D_t("DataView", v_r.extent(0));
    ViewDevice_1D_t v_d_out_3 = ViewDevice_1D_t("DataView", v_r.extent(0));
    ViewHost_1D_t v_h = ViewHost_1D_t("HostView", v_r.extent(0));

    int num_teams = L;
    int num_teams_adjusted = num_teams - 2;
    int team_size = M;
    int thread_vector_length = 1;
    int next_rank = (myRank + 1) % numRanks;

    auto remote_range =
        Kokkos::Experimental::get_range(dim0, (myRank + 1) % numRanks);
    auto local_range = Kokkos::Experimental::get_range(dim0, myRank);
    int size_per_rank = remote_range.second - remote_range.first + 1;
    printf("Range[%i,%i]\n", remote_range.first, remote_range.second);
    int size_per_team = size_per_rank / num_teams_adjusted;
    int size_per_team_mod = size_per_rank % num_teams_adjusted;

    auto policy =
        Kokkos::TeamPolicy<>(num_teams, team_size, thread_vector_length);
    using team_t = Kokkos::TeamPolicy<>::member_type;

    Kokkos::parallel_for(
        "Init", size_per_rank,
        KOKKOS_LAMBDA(const int i) { v_d(i) = local_range.first + i; });

    Kokkos::fence();
    RemoteSpace_t().fence();

    Kokkos::Experimental::remote_parallel_for(
        "Increment", policy,
        KOKKOS_LAMBDA(const team_t &team) {
          int start = team.league_rank() * size_per_team;
          int block = team.league_rank() == team.league_size() - 1
                          ? size_per_team + size_per_team_mod
                          : size_per_team;

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, block),
                               [&](const int i) {
                                 int index = remote_range.first + start + i;
                                 //   printf("i:%i\n", index);
                                 v_d_out_1(start + i) = v_r(index);
                                 /*v_d_out_2(start + i) = v_r(index);
                                   v_d_out_3(start + i) = v_r(index);*/
                               });
        },
        v_r);

    Kokkos::fence();
    RemoteSpace_t().fence();

    Kokkos::deep_copy(v_h, v_d_out_1);
    for (int i = 0; i < size_per_rank; ++i)
      assert(v_h(i) == next_rank * size_per_rank + i);
  }
  Kokkos::finalize();
#ifdef KRS_ENABLE_NVSHMEMSPACE
  nvshmem_finalize();
#endif
#ifdef KRS_ENABLE_SHMEMSPACE
  shmem_finalize();
#else
  MPI_Finalize();
#endif

  return 0;
}
