/*!
 * \file   src/dune-mgis-tensile-test.cc
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

constexpr size_type getStrainSize(const size_type d) {
  if (d == 2) {
    return 4;
  } else if (d != 3) {
    throw std::logic_error("getStrainSize: unsupported dimension");
  }
  return 6;
}  // end of getStrainSize

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
  constexpr const auto g_stride = getStrainSize(dime);
  constexpr const auto th_stride = getStrainSize(dime);
  constexpr const auto K_stride = getStrainSize(dime) * getStrainSize(dime);
  constexpr const auto cste = double(1.41421356237309504880);
  const auto& e = lv.element();
  const auto& lfem = lv.tree().finiteElement();
  const auto update_strain = [&cste](double* const e,  //
                                      const Dune::FieldVector<double, dime>& g,
                                      const Dune::FieldVector<double, dime>& u) {
    if constexpr(dime == 3) {
        // strain expressed in vector notations using MFront conventions
        e[0] += u[0] * g[0];
        e[1] += u[1] * g[1];
        e[2] += u[2] * g[2];
        e[3] += cste * (u[0] * g[1] + u[1] * g[0]) / 2;
        e[4] += cste * (u[0] * g[2] + u[2] * g[0]) / 2;
        e[5] += cste * (u[2] * g[1] + u[1] * g[2]) / 2;
    } else {
      std::runtime_error e(
          "assembleElementInnerForcesAndStiffnessMatrix: "
          "unsupported space dimension");
      throw(e);
    }
  };  // end of update_strain
  const auto get_strain_matrix = [&cste](const Dune::FieldVector<double, dime>& g) {
    auto B = Dune::FieldMatrix<double, g_stride, dime> {};
    B = 0;
    if constexpr(dime == 3) {
        // exx = dux/dx
        B[0][0] = g[0];
        // eyy = duy/dy
        B[1][1] = g[1];
        // ezz = duz/dz
        B[2][2] = g[2];
        // exy = sqrt(2)*(dux/dy+duy/dx)/2
        B[3][0] = cste * g[1] / 2;
        B[3][1] = cste * g[0] / 2;
        // exz = sqrt(2)*(dux/dz+duz/dx)/2
        B[4][0] = cste * g[2] / 2;
        B[4][2] = cste * g[0] / 2;
        // eyz = sqrt(2)*(duz/dy+duy/dz)/2
        B[5][1] = cste * g[2] / 2;
        B[5][2] = cste * g[1] / 2;
    } else {
      std::runtime_error e(
          "assembleElementInnerForcesAndStiffnessMatrix: "
          "unsupported space dimension");
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
    // poids
    const auto w = ip.weight();
    // Compute the strain
    double* const e0 = m.s0.gradients.data() + g_stride * ip_id;
    double* const e1 = m.s1.gradients.data() + g_stride * ip_id;
    std::fill(e0, e0 + g_stride, 0);
    std::fill(e1, e1 + g_stride, 0);
    // loop over the nodes
    for (size_type i = 0; i < referenceGradients.size(); ++i) {
      constexpr const auto cste = double{1.41421356237309504880};
      Dune::FieldVector<double, dime> g;
      jacobian.mv(referenceGradients[i][0], g);
      // displacement of the ith local node
      auto nid = lv.index(i);
      // updating the strain
      update_strain(e0, g, u0[nid]);
      update_strain(e1, g, u1[nid]);
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
    //     double s2[6];
    //     const auto young = double{150e9};
    //     const auto nu = double{0.3};
    //     const auto lambda = nu * young / ((1 + nu) * (1 - 2 * nu));
    //     const auto mu = young / (2 * (1 + nu));
    //     const auto tr = e1[0] + e1[1] + e1[2];
    //     s2[0] = lambda * tr + 2 * mu * e1[0];
    //     s2[1] = lambda * tr + 2 * mu * e1[1];
    //     s2[2] = lambda * tr + 2 * mu * e1[2];
    //     s2[3] = 2 * mu * e1[3];
    //     s2[4] = 2 * mu * e1[4];
    //     s2[5] = 2 * mu * e1[5];
    //     std::cout << "stresses at: " << ip_id << " " << m.s1.thermodynamic_forces.size() /
    //     th_stride
    //               << '\n';
    //     for (size_type i = 0; i != 6; ++i) {
    //       std::cout << s1[i] << " " << s2[i] << " " << std::abs(s1[i] - s2[i]) << '\n';
    //     }
    std::copy(s1, s1 + th_stride, sig.begin());
    // std::copy(s2, s2 + th_stride, sig.begin());
    //     // the consistent tangent operator
    auto Ke = Dune::FieldMatrix<double, th_stride, g_stride>{};
    const auto* const K1 = m.K.data() + K_stride * ip_id;
    for (size_type i = 0; i < th_stride; ++i) {
      for (size_type j = 0; j < g_stride; ++j) {
        Ke[i][j] = K1[g_stride * i + j];
      }
    }
    //     Ke = 0;
    //     for (size_type i = 0; i != 3; ++i) {
    //       for (size_type j = 0; j != 3; ++j) {
    //         Ke[i][j] = lambda;
    //       }
    //     }
    //     for (size_type i = 0; i != 6; ++i) {
    //       Ke[i][i] += 2 * mu;
    //     }
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
  for (size_type i = 0; i < K.N(); i++) {
    if (dirichletNodes[i]) {
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

void tensile_test(const std::string& behaviour) {
  // space dimension
  constexpr const auto dime = size_type{3};
  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, dime, dime>>;
  using Vector = Dune::BlockVector<Dune::FieldVector<double, dime>>;
  // type aliases
  using Grid = Dune::YaspGrid<dime>;
  using GridView = Grid::LeafGridView;
  /* build a simple unit cube in 3D */
  // Lenghts of the grid
  Dune::FieldVector<double, 3u> L{1.0, 1.0, 1.0};
  // number of element of the grid
  std::array<size_type, 3u> N{1, 1, 1};
  // building the grid
  std::cout << "start building the grid" << std::endl;
   auto g = std::make_shared<Grid>(L, N);
   std::cout << "end building the grid" << std::endl;
   // taking the leaf of the grid
   auto gv = g->leafGridView();
   /* finite element basis */
   auto basis = Dune::Functions::LagrangeBasis<GridView, 1>(gv);
   // loading the behaviour
   const auto b = mgis::behaviour::load("libBehaviour.so", behaviour,
                                        mgis::behaviour::Hypothesis::TRIDIMENSIONAL);
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
   auto times = std::vector<double>(10);
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
   //    Dune::VTKSequenceWriter<GridView> vtkSequenceWriter(vtkWriter, "results-" + behaviour);
   //    vtkSequenceWriter.addVertexData(u1, "displacement", dime);
   /** loop over the time steps **/
   auto nstep = size_type{};
   for (; ptdt != times.end(); ++pt, ++ptdt, ++nstep) {
     // imposed displacement
     const auto u = 1.e-2 * (*ptdt / times.back());
     std::cout << "* time step from " << *pt << " to " << *ptdt << std::endl;
     /** simulate a newton with a fixed number of iterations **/
     for (size_type n = 0; n != 10; ++n) {
       if (n != 0) {
         // copy the state of the material at the beginning of the time step on the
         // material state at the end of the time step
         mgis::behaviour::revert(m);
       }
       // building the inner forces of the stiffness matrix
       buildInnerForcesAndStiffnessMatrix(K, r, m, basis, u0, u1, *ptdt - *pt);
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
       //      std::cout << "nf: " << nf << '\n';
       // apply Dirichet boundary conditions
       const auto eps = 1.e-10;
       // ux = 0 if x==0
       apply_boundary_conditions(K, r, basis, u1, nf, 0,
                                 [eps](const auto& x) { return std::abs(x[0]) < eps; });
       // uy = 0 if y==0
       apply_boundary_conditions(K, r, basis, u1, nf, 1,
                                 [eps](const auto& x) { return std::abs(x[1]) < eps; });
       // uz = 0 if z==0
       apply_boundary_conditions(K, r, basis, u1, nf, 2,
                                 [eps](const auto& x) { return std::abs(x[2]) < eps; });
       // uz = u if z==1
       apply_boundary_conditions(K, r, basis, u1, nf, 2,
                                 [eps](const auto& x) { return std::abs(x[2] - 1) < eps; }, u);
       ///////////////////////////
       //   Compute solution
       ///////////////////////////

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
       //
       u1 -= du;
     }
     u0 = u1;
     // update the state of the material
     mgis::behaviour::update(m);
     // strain
     const auto eto = m.s1.gradients.data();
     // stress
     const auto sig = m.s1.thermodynamic_forces.data();
     out << *ptdt << " " << u << " "                         //
         << eto[0] << " " << eto[1] << " " << eto[2] << " "  //
         << eto[3] << " " << eto[4] << " " << eto[5] << " "  //
         << sig[0] << " " << sig[1] << " " << sig[2] << " "  //
         << sig[3] << " " << sig[4] << " " << sig[5] << "}\n";

     //
     //      const auto uh =
     //      Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, dime>>(
     //              vtk_basis, u1);
     //     vtkSequenceWriter.write(*ptdt);

     //      auto vtkWriter = Dune::VTKWriter<GridView>(gv);
     //      const auto uh =
     //          Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, dime>>(
     //              vtk_basis, u);
     //      vtkWriter.addVertexData(
     //          uh, Dune::VTK::FieldInfo("displacement", Dune::VTK::FieldInfo::Type::vector,
     //          dime));
     //      vtkWriter.write("dune-mgis-"+behaviour+"-" + std::to_string(nstep));
   }
}  // end of tensile_test

void elasticity_test() { tensile_test("Elasticity"); }

void plasticity_test() { tensile_test("Plasticity"); }

int main(int argc, char** argv) {
  try {
    // Maybe initialize MPI
    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    elasticity_test();
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
