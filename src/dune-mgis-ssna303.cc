/*!
 * \file   src/dune-mgis-ssna303.cc
 * \brief
 * \author Thomas Helfer
 * \date   07/04/2020
 */

#include <iostream>
#include <algorithm>
#include <memory>
#include <cstdlib>

#include "config.h"
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/exceptions.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/functions/functionspacebases/basistags.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/istl/matrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/superlu.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
//#include <dune/grid/io/file/vtk/vtksequencewriter.hh>

#include "MGIS/Behaviour/Behaviour.hxx"
#include "MGIS/Behaviour/MaterialDataManager.hxx"
#include "MGIS/Behaviour/Integrate.hxx"

using size_type = int;

//! \brief matrice-vector product
template <typename real, size_type N, size_type M>
Dune::FieldVector<real, N> mmult(const Dune::FieldMatrix<real, N, M>& m,
                                 const Dune::FieldVector<real, M>& v) {
  auto r = Dune::FieldVector<real, N> {};
  for (size_type i= 0; i != N; ++i) {
    r[i] = 0;
    for (size_type j= 0; j != M; ++j) {
      r[i] += m[i][j] * v[j];
    }
  }
  return r;
}  // end of mmult

//! \brief multiply two matrices
template <typename real, size_type N, size_type M, size_type L>
Dune::FieldMatrix<real, N, M> mmult(const Dune::FieldMatrix<real, N, L>& a,
                                    const Dune::FieldMatrix<real, L, M>& b) {
  auto c = Dune::FieldMatrix<real, N, M> {};
  for (size_type i= 0; i != N; ++i) {
    for (size_type j= 0; j != M; ++j) {
      c[i][j] = 0;
      for (size_type k = 0; k != L; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}  // end of mmult

//! \brief transpose a matrix
template <typename real, size_type N, size_type M>
Dune::FieldMatrix<real, M, N> mtrans(const Dune::FieldMatrix<real, N, M>& m) {
  auto r = Dune::FieldMatrix<real, M, N>{};
  for (size_type i= 0; i != M; ++i) {
    for (size_type j= 0; j != N; ++j) {
      r[i][j] = m[j][i];
    }
  }
  return r;
} // end of mtrans


/*!
 * \return the number of integration points for the given order
 * of quadrature.
 * \tparam dime: space dimension
 * \tparam Base: type of the finite element basis
 * \param[in] b: finite element basis
 * \param[in] o: order of quadrature
 */
template <size_type dime, typename Basis>
size_type getNumberOfIntegrationPoints(const Basis& b, const size_type o) {
  const auto& gv = b.gridView();
  auto n = size_type{};
  for (const auto& e : elements(gv)) {
    const auto& q = Dune::QuadratureRules<double, dime>::rule(e.type(), o);
    n += q.size();
  }
  return n;
}  // end of getNumberOfIntegrationPoints

/*!
 * \return the number of integration points for the default order
 * of quadrature.
 * \tparam dime: space dimension
 * \tparam Base: type of the finite element basis
 * \param[in] b: finite element basis
 */
template <size_type dime, typename Basis>
size_type getNumberOfIntegrationPoints(const Basis& b) {
  auto n = size_type{};
  const auto& gv = b.gridView();
  auto lv = b.localView();
  for (const auto& e : elements(gv)) {
    lv.bind(e);
    const auto& le = lv.tree().finiteElement();
    const auto o = 2 * le.localBasis().order();
    const auto& q = Dune::QuadratureRules<double, dime>::rule(e.type(), o);
    n += q.size();
  }
  return n;
}  // end of getNumberOfIntegrationPoints

/*!
 * \brief initialize the stiffness matrix
 */
template <typename Basis>
Dune::BCRSMatrix<Dune::FieldMatrix<double, Basis::GridView::dimension, Basis::GridView::dimension>>
initializeStiffnessMatrix(const Basis& b) {
  constexpr const auto dime = Basis::GridView::dimension;
  Dune::MatrixIndexSet nb;
  nb.resize(b.size(), b.size());
  // A loop over all elements of the grid
  auto lv = b.localView();
  for (const auto& e : elements(b.gridView())) {
    lv.bind(e);
    for (size_type i = 0; i != lv.size(); ++i) {
      const auto r = lv.index(i);
      for (size_type j = 0; j != lv.size(); ++j) {
        const auto c = lv.index(j);
        nb.add(r, c);
      }
    }
  }
  auto K = Dune::BCRSMatrix<Dune::FieldMatrix<double, dime, dime>>{};
  nb.exportIdx(K);
  return K;
}  // end of initializeStiffnessMatrix

constexpr size_type getDeformationGradientSize(const size_type d) {
  if (d == 2) {
    return 5;
  } else if (d != 3) {
    throw std::logic_error("getDeformationGradientSize: unsupported dimension");
  }
  return 9;
}  // end of getDeformationGradientSize

template <typename Matrix, typename Vector, typename LocalView>
// std::tuple<Dune::Matrix<double>, Dune::BlockVector<double>>
void assembleElementInnerForcesAndStiffnessMatrix(Matrix& K,
                                                  Vector& r,
                                                  mgis::behaviour::MaterialDataManager& m,
                                                  mgis::size_type& ip_id,
                                                  const LocalView& lv,
                                                  const Vector& u0,
                                                  const Vector& u1,
                                                  const double dt) {
  using Element = typename LocalView::Element;
  constexpr const auto dime = Element::dimension;
  constexpr const auto g_stride = getDeformationGradientSize(dime);
  constexpr const auto th_stride = getDeformationGradientSize(dime);
  constexpr const auto K_stride = getDeformationGradientSize(dime) * getDeformationGradientSize(dime);
  constexpr const auto cste = double(1.41421356237309504880);
  const auto& e = lv.element();
  const auto& lfem = lv.tree().finiteElement();
  const auto update_deformation_gradient = [&cste](double* const F,  //
                                                   const Dune::FieldVector<double, dime>& g,
                                                   const Dune::FieldVector<double, dime>& u) {
    if
      constexpr(dime == 3) {
        // strain expressed in vector notations using MFront conventions
        F[0] += u[0] * g[0];
        F[1] += u[1] * g[1];
        F[2] += u[2] * g[2];
        F[3] += u[0] * g[1];
        F[4] += u[1] * g[0];
        F[5] += u[0] * g[2];
        F[6] += u[2] * g[0];
        F[7] += u[1] * g[2];
        F[8] += u[2] * g[1];
      }
    else if
      constexpr(dime == 2) {
        // plane strain
        F[0] += u[0] * g[0];
        F[1] += u[1] * g[1];
        F[3] += u[0] * g[1];
        F[4] += u[1] * g[0];
      }
    else {
      std::runtime_error e(
          "assembleElementInnerForcesAndStiffnessMatrix: "
          "unsupported space dimension (" +
          std::to_string(dime) + ")");
      throw(e);
    }
  };  // end of update_deformation_gradient
  const auto get_strain_matrix = [](const Dune::FieldVector<double, dime>& g) {
    auto B = Dune::FieldMatrix<double, g_stride, dime> {};
    B = 0;
    if constexpr(dime == 3) {
        // dux/dx
        B[0][0] = g[0];
        // duy/dy
        B[1][1] = g[1];
        // duz/dz
        B[2][2] = g[2];
        // dux_dy
        B[3][0] = g[1] ;
        // duy_dx
        B[4][1] = g[0] ;
        // dux_dz
        B[5][0] = g[2] ;
        // duz_dx
        B[6][2] = g[0] ;
        // duy_dz
        B[7][1] = g[2] ;
        // duz_dy
        B[8][2] = g[1] ;
      }
    else if
      constexpr(dime == 2) {
        // dux/dx
        B[0][0] = g[0];
        // duy/dy
        B[1][1] = g[1];
        // dux_dy
        B[3][0] = g[1] ;
        // duy_dx
        B[4][1] = g[0] ;
    } else {
      std::runtime_error e(
          "assembleElementInnerForcesAndStiffnessMatrix: "
          "unsupported space dimension (" +
          std::to_string(dime) + ")");
      throw(e);
    }
    return B;
  }; // end of get_strain_matrix
  // assertions
  assert(m.s0.gradients_stride == g_stride);
  assert(m.s1.gradients_stride == g_stride);
  assert(m.s1.thermodynamic_forces_stride == th_stride);
  assert(m.K_stride == K_stride);
  // std::cout << "assembleElementInnerForcesAndStiffnessMatrix: begin\n";
  const auto order = 2 * lfem.localBasis().order();
  // loop over the integration points
  for (const auto& ip : Dune::QuadratureRules<double, dime>::rule(e.type(), order)) {
    // std::cout << "treating integration point number " << ip_id << '\n';
    // Position of the current quadrature point in the reference element
    const auto x = ip.position();
    // The gradients of the shape functions on the reference element
    std::vector<Dune::FieldMatrix<double, 1, dime>> referenceGradients;
    lfem.localBasis().evaluateJacobian(x, referenceGradients);
    // The transposed inverse Jacobian of the map from the reference element to the element
    const auto jacobian = e.geometry().jacobianInverseTransposed(x);
    // The determinant term in the integral transformation formula
    const auto J = e.geometry().integrationElement(x);
    // weight of the integration point
    const auto w = ip.weight();
    // Compute the strain
    double* const F0 = m.s0.gradients.data() + g_stride * ip_id;
    double* const F1 = m.s1.gradients.data() + g_stride * ip_id;
    std::fill(F0, F0 + g_stride, 0);
    F0[0] = F0[1] = F0[2] = 1;
    std::fill(F1, F1 + g_stride, 0);
    F1[0] = F1[1] = F1[2] = 1;
    // loop over the nodes
    for (size_type i = 0; i < referenceGradients.size(); ++i) {
      Dune::FieldVector<double, dime> g;
      jacobian.mv(referenceGradients[i][0], g);
      // displacement of the ith local node
      auto nid = lv.index(i);
      // updating the strain
      update_deformation_gradient(F0, g, u0[nid]);
      update_deformation_gradient(F1, g, u1[nid]);
    }
    // integrate the behaviour
    const auto tg_opt = mgis::behaviour::IntegrationType::INTEGRATION_CONSISTENT_TANGENT_OPERATOR;
    if (mgis::behaviour::integrate(m, tg_opt, dt, ip_id, ip_id + 1) != 1) {
      std::runtime_error e("integration failed");
      throw(e);
    }
    //     // the stresses
    auto sig = Dune::FieldVector<double, th_stride>{};
    const auto* const s1 = m.s1.thermodynamic_forces.data() + th_stride * ip_id;
    std::copy(s1, s1 + th_stride, sig.begin());
    // the consistent tangent operator
    auto Ke = Dune::FieldMatrix<double, th_stride, g_stride>{};
    const auto* const K1 = m.K.data() + K_stride * ip_id;
    for (size_type i = 0; i < th_stride; ++i) {
      for (size_type j = 0; j < g_stride; ++j) {
        Ke[i][j] = K1[g_stride * i + j];
      }
    }
    // loop of the nodes
    for (size_type i = 0; i != referenceGradients.size(); ++i) {
      auto ri = lv.index(i);
      Dune::FieldVector<double, dime> gi;
      jacobian.mv(referenceGradients[i][0], gi);
      const auto tBi = mtrans(get_strain_matrix(gi));
      auto re = Dune::FieldVector<double, dime>{};
      r[ri] += w * J * mmult(tBi, sig);
      for (size_type j = 0; j != referenceGradients.size(); ++j) {
        auto rj = lv.index(j);
        Dune::FieldVector<double, dime> gj;
        jacobian.mv(referenceGradients[j][0], gj);
        const auto Bj = get_strain_matrix(gj);
        K[ri][rj] += w * J * mmult(mmult(tBi, Ke), Bj);
      }
    }
    ++ip_id;
  }  // loop over quadrature points
  //  std::cout << "assembleElementInnerForcesAndStiffnessMatrix: end\n";
}  // end of assembleElementInnerForcesAndStiffnessMatrix

template <typename Matrix, typename Vector, typename Basis>
void buildInnerForcesAndStiffnessMatrix(Matrix& K,
                                        Vector& r,
                                        mgis::behaviour::MaterialDataManager& m,
                                        const Basis& basis,
                                        const Vector& u0,
                                        const Vector& u1,
                                        const double dt) {
  constexpr const auto dime = Basis::GridView::dimension;
  auto lv = basis.localView();
  auto ip_id = mgis::size_type{};
  auto ne = size_type{};
  K = 0;
  r = 0;
  for (const auto& e : elements(basis.gridView())) {
    lv.bind(e);
    assembleElementInnerForcesAndStiffnessMatrix(K, r, m, ip_id, lv, u0, u1, dt);
    ++ne;
  }
}  // end of buildInnerForcesAndStiffnessMatrix

template <typename Matrix, typename Vector, typename Basis, typename Predicate>
void apply_boundary_conditions(Matrix& K,
                               Vector& r,
                               const Basis& b,
                               const Vector& u1,
                               const double nf,
                               const size_type c,
                               const Predicate& p,
                               const double u = 0) {
  // Evaluating the predicate will mark all Dirichlet degrees of freedom
  std::vector<bool> dirichletNodes;
  Dune::Functions::interpolate(b, dirichletNodes, p);
  ///////////////////////////////////////////
  //   Modify Dirichlet rows
  ///////////////////////////////////////////
  // Loop over the matrix rows
  auto n = size_type{};
  for (size_type i = 0; i < K.N(); i++) {
    if (dirichletNodes[i]) {
      ++n;
      auto cIt = K[i].begin();
      auto cEndIt = K[i].end();
      // Loop over nonzero matrix entries in current row
      for (; cIt != cEndIt; ++cIt){
        auto& Ke = *cIt;
        for (size_type j = 0; j != Ke.M(); ++j){
          Ke[c][j] = 0;
        }
        if (cIt.index() == i) {
          Ke[c][c] = nf;
        }
      }
      r[i][c] = nf * (u1[i][c] - u);
    }
  }
}

void ssna303_test(const std::string& behaviour) {
  // space dimension
  constexpr const auto dime = size_type{2};
  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, dime, dime>>;
  using Vector = Dune::BlockVector<Dune::FieldVector<double, dime>>;
  // type aliases
//  using Grid = Dune::YaspGrid<dime>;
  using Grid = Dune::UGGrid<dime>;
  using GridView = Grid::LeafGridView;
  // reading the grid
  auto g = Dune::GmshReader<Grid>::read("ssna303-mesh.msh", true, false);
//  Dune::FieldVector<double, 2u> L{4.8e-3, 30e-3};
//  std::array<size_type, 2u> N{1, 1};
//  auto g = std::make_shared<Grid>(L, N);
  // taking the leaf of the grid
  auto gv = g->leafGridView();
  /* finite element basis */
  auto basis = Dune::Functions::LagrangeBasis<GridView, 1>(gv);
  // loading the behaviour
  auto opts = mgis::behaviour::FiniteStrainBehaviourOptions{};
  opts.stress_measure = mgis::behaviour::FiniteStrainBehaviourOptions::PK1;
  opts.tangent_operator = mgis::behaviour::FiniteStrainBehaviourOptions::DPK1_DF;
  const auto b = mgis::behaviour::load(opts, "libBehaviour.so", behaviour,
                                       mgis::behaviour::Hypothesis::PLANESTRAIN);
  // initialize the material manager. In this example it manages all the memory required by the
  // behaviour.
  auto m = mgis::behaviour::MaterialDataManager(b, getNumberOfIntegrationPoints<dime>(basis));
  //    // The behaviours tested have no material properties,
  //    // we only set the temperature to be uniform
  setExternalStateVariable(m.s0, "Temperature", 293.15);
  setExternalStateVariable(m.s1, "Temperature", 293.15);
  // initialize the stiffness matrix
  auto K = initializeStiffnessMatrix(basis);
  // initialize the residual
  auto r = Vector{};
  r.resize(basis.size());
  // displacement fields
  auto u0 = Vector{};
  u0.resize(basis.size());
  u0 = 0;
  auto u1 = Vector{};
  u1.resize(basis.size());
  u1 = 0;
  //
  const auto eps = 1.e-10;
  std::vector<bool> upper_nodes;
  Dune::Functions::interpolate(basis, upper_nodes,
                               [eps](const auto& x) { return std::abs(x[1] - 3.00000e-02) < eps; });
  //
  auto times = std::vector<double>(50);
  for (size_type i = 0; i != times.size(); ++i) {
    times[i] = double(i) / (times.size() - 1);
   }
   //
   auto pt = times.begin();
   auto ptdt = std::next(pt);
   auto out = std::ofstream("results-" + behaviour + ".txt");
   out.precision(15);
   //     Dune::VTKWriter<GridView> vtkWriter(gv);
   const auto vtk_basis = Dune::Functions::BasisFactory::makeBasis(
       gv, Dune::Functions::BasisFactory::power<dime>(
               Dune::Functions::BasisFactory::lagrange<1>(),
               Dune::Functions::BasisFactory::blockedInterleaved()));
   auto nstep = size_type{};
   auto Fi = double{};
   /** loop over the time steps **/
   for (; ptdt != times.end(); ++pt, ++ptdt) {
     // imposed displacement
     const auto u = 6.e-3 * (*ptdt / times.back());
     std::cout << "* time step from " << *pt << " to " << *ptdt << std::endl;
     /** simulate a newton with a fixed number of iterations **/
     for (size_type n = 0; n != 20; ++n) {
       if (n != 0) {
         // copy the state of the material at the beginning of the time step on the
         // material state at the end of the time step
         mgis::behaviour::revert(m);
       }
       // building the inner forces of the stiffness matrix
       buildInnerForcesAndStiffnessMatrix(K, r, m, basis, u0, u1, *ptdt - *pt);
       // computing the forces
       Fi = double{};
       for (size_type i = 0; i < K.N(); i++) {
         if (upper_nodes[i]) {
           Fi += r[i][1];
         }
       }
       // normalisation factor
       const auto nf = [&K] {
         auto v = double(-1);
         for (size_type n = 0; n < K.N(); n++) {
           auto cIt = K[n].begin();
           auto cEndIt = K[n].end();
           // Loop over nonzero matrix entries in current row
           for (; cIt != cEndIt; ++cIt) {
             auto& Ke = *cIt;
             for (size_type i = 0; i != Ke.M(); ++i) {
               for (size_type j = 0; j != Ke.M(); ++j) {
                 v = std::max(v, std::abs(Ke[i][j]));
               }
             }
           }
         }
         return v;
       }();
       // apply Dirichet boundary conditions
       // ux = 0 if x==0
       apply_boundary_conditions(K, r, basis, u1, nf, 0,
                                 [eps](const auto& x) { return std::abs(x[0]) < eps; });
       // uy = 0 if y==0
       apply_boundary_conditions(K, r, basis, u1, nf, 1,
                                 [eps](const auto& x) { return std::abs(x[1]) < eps; });
       // uy = u if y==l
       apply_boundary_conditions(
           K, r, basis, u1, nf, 1,
           [eps](const auto& x) { return std::abs(x[1] - 3.00000e-02) < eps; }, u);
       ///////////////////////////
       //   Compute solution
       ///////////////////////////

       const auto rsum = std::accumulate(r.begin(), r.end(), Dune::FieldVector<double, 2>(0));
       std::cout << "iter: " << n << " " << abs(rsum[0]) + abs(rsum[1]) << '\n';

       // opposite of the correction to the displacement field
       Vector du(basis.size());
       du = r;

       Dune::InverseOperatorResult statistics;
       //       // Turn the matrix into a linear operator
       //       Dune::MatrixAdapter<Matrix, Vector, Vector> linearOperator(K);
       //
       //       // Sequential incomplete LU decomposition as the preconditioner
       //       Dune::SeqILU<Matrix, Vector, Vector> preconditioner(K, 1.0);

       // Preconditioned conjugate gradient solver
       //       Dune::CGSolver<Vector> cg(linearOperator, preconditioner,
       //                                 1e-5,  // Desired residual reduction factor
       //                                 50,    // Maximum number of iterations
       //                                 2);    // Verbosity of the solver
       // Object storing some statistics about the solving process
       // Solve!
       //       cg.apply(du, r, statistics);

       Dune::SuperLU<Matrix> lu_solver(K);
       lu_solver.apply(du, r, statistics);
       if (!statistics.converged) {
         throw(std::runtime_error("linear system resolution failed"));
       }
       //
       u1 -= du;
     }
     u0 = u1;
     // update the state of the material
     mgis::behaviour::update(m);
     out << u << " " << Fi << '\n';
     //      auto vtkWriter = Dune::VTKWriter<GridView>(gv);
     //      const auto uh =
     //          Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, dime>>(
     //              vtk_basis, u);
     //      vtkWriter.addVertexData(
     //          uh, Dune::VTK::FieldInfo("displacement", Dune::VTK::FieldInfo::Type::vector,
     //          dime));
     //      vtkWriter.write("dune-mgis-" + behaviour + "-" + std::to_string(nstep));
   }
   out.close();
}  // end of ssna303_test

void plasticity_test() { ssna303_test("LogarithmicStrainPlasticity"); }

int main(int argc, char** argv) {
  try {
    // Maybe initialize MPI
    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    plasticity_test();
  } catch (Dune::Exception& e) {
    std::cerr << "Dune reported error: " << e << std::endl;
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    std::cerr << "Standard exception caught: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "Unknown exception thrown!" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
