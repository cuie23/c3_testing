#include "iC3.h"

#include <cmath>

#include <Eigen/Dense>

#include "core/c3_miqp.h"
#include "core/c3_plus.h"
#include "core/c3_qp.h"
#include "multibody/lcs_factory.h"
#include "common/quaternion_error_hessian.h"

#include "drake/common/text_logging.h"

using drake::multibody::ModelInstanceIndex;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::DiscreteValues;
using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::VectorXd;
using Eigen::VectorXf;

namespace c3 {
namespace systems {

iC3::iC3(
    drake::multibody::MultibodyPlant<double>& plant,
    drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    C3::CostMatrices& costs, 
    C3ControllerOptions controller_options)
    : plant_(plant),
      plant_ad_(plant_ad),
      controller_options_(controller_options),
      N_(controller_options_.lcs_factory_options.N) {
  this->set_name("iC3");

  // Initialize dimensions
  n_q_ = plant_.num_positions();
  n_v_ = plant_.num_velocities();
  n_u_ = plant_.num_actuators();
  n_x_ = n_q_ + n_v_;
  dt_ = controller_options_.lcs_factory_options.dt;

  // Determine the size of lambda based on the contact model
  n_lambda_ = multibody::LCSFactory::GetNumContactVariables(
      controller_options_.lcs_factory_options);

  // Placeholder vector for initialization
  VectorXd zeros = VectorXd::Zero(n_x_ + n_lambda_ + n_u_);

  // Create placeholder LCS and desired state for the base C3 problem
  auto lcs_placeholder =
      LCS::CreatePlaceholderLCS(n_x_, n_u_, n_lambda_, N_, dt_);
  auto x_desired_placeholder =
      std::vector<VectorXd>(N_ + 1, VectorXd::Zero(n_x_));

  // Initialize the C3 problem based on the projection type
  if (controller_options_.projection_type == "MIQP") {
    c3_ =
        std::make_unique<C3MIQP>(lcs_placeholder, costs, x_desired_placeholder,
                                 controller_options_.c3_options);
  } else if (controller_options_.projection_type == "QP") {
    c3_ = std::make_unique<C3QP>(lcs_placeholder, costs, x_desired_placeholder,
                                 controller_options_.c3_options);
  } else if (controller_options_.projection_type == "C3+") {
    c3_ =
        std::make_unique<C3Plus>(lcs_placeholder, costs, x_desired_placeholder,
                                 controller_options_.c3_options);
  } else {
    drake::log()->error("Unknown projection type : {}",
                        controller_options_.projection_type);
  }
  DRAKE_THROW_UNLESS(c3_ != nullptr);

}

  pair<vector<MatrixXd>, vector<MatrixXd>> iC3::ComputeTrajectory(
    drake::systems::Context<double>& context,
    drake::systems::Context<drake::AutoDiffXd>& context_ad, 
    const std::vector<drake::SortedPair<drake::geometry::GeometryId>>& contact_geoms) {

    std::vector<double> x_init = *controller_options_.x_init;
    VectorXd x0 = Eigen::Map<VectorXd>(x_init.data(), x_init.size());    

    std::vector<double> x_des = *controller_options_.x_des;
    VectorXd xd = Eigen::Map<VectorXd>(x_des.data(), x_des.size());  

    // ith column = ith timestep
    MatrixXd x_hat = x0.replicate(1, N_+1);
    MatrixXd u_hat(Eigen::MatrixXd::Zero(n_u_, N_));

    int num_iters = 50;

    vector<MatrixXd> all_x_hats;
    vector<MatrixXd> all_u_hats;

    all_x_hats.push_back(x_hat);
    all_u_hats.push_back(u_hat);

    LCSFactory lcs_factory(plant_, context, plant_ad_, context_ad, 
        contact_geoms, controller_options_.lcs_factory_options);

    lcs_factory.UpdateStateAndInput(x0, u_hat.col(0));
    LCS lcs = lcs_factory.GenerateLCS();

    for (int iter = 0; iter < num_iters-1; iter++) {
      
      std::cout << "iC3 iteration " << iter << std::endl;
      UpdateQuaternionCosts(x_hat, xd);
      C3::CostMatrices new_costs(Q_, R_, G_, U_);
      c3_->UpdateCostMatrices(new_costs);

      std::vector<VectorXd> target =
        std::vector<VectorXd>(N_ + 1, xd);
      c3_->UpdateLCS(lcs);
      c3_->UpdateTarget(target);
      c3_->Solve(x_hat.col(0));

      //vector<Eigen::VectorXd> x_sol = c3_->GetStateSolution();
      vector<Eigen::VectorXd> u_sol = c3_->GetInputSolution();

      for (int k = 0; k < N_; k++) {
        //x_hat.col(k) = x_sol[k];
        u_hat.col(k) = u_sol[k];
      }

      auto output = DoLCSRollout(x0, u_hat, lcs_factory);
      lcs = output.first;
      x_hat = output.second;


      all_x_hats.push_back(x_hat);
      all_u_hats.push_back(u_hat);

    }
    std::cout << "returned all_x_hats" << std::endl;
    return std::make_pair(all_x_hats, all_u_hats);
  }


  pair<LCS, MatrixXd> iC3::DoLCSRollout(VectorXd x0, MatrixXd u_hat, LCSFactory factory) {

    // Set up time varying LCS
    vector<Eigen::MatrixXd> A;
    vector<Eigen::MatrixXd> B;
    vector<Eigen::MatrixXd> D;
    vector<Eigen::VectorXd> d;
    vector<Eigen::MatrixXd> E;
    vector<Eigen::MatrixXd> F;
    vector<Eigen::MatrixXd> H;
    vector<Eigen::VectorXd> c;
    A.clear();
    B.clear();
    D.clear();
    d.clear();
    E.clear();
    F.clear();
    H.clear();
    c.clear();

    MatrixXd x_hat(x0.size(), N_+1);
    x_hat.col(0) = x0;
    VectorXd x_curr = x0;
    VectorXd x_next;

    for (int k = 0; k < N_; k++) {

      // Linearize about current point
      factory.UpdateStateAndInput(x_curr, u_hat.col(k));
      LCS lcs = factory.GenerateLCS();
      A.push_back(lcs.A()[0]);
      B.push_back(lcs.B()[0]);
      D.push_back(lcs.D()[0]);
      d.push_back(lcs.d()[0]);
      E.push_back(lcs.E()[0]);
      F.push_back(lcs.F()[0]);
      H.push_back(lcs.H()[0]);
      c.push_back(lcs.c()[0]);      

      // Do one rollout step
      VectorXd u_k = u_hat.col(k);
      x_next = lcs.Simulate(x_curr, u_k, true);
      x_hat.col(k+1) = x_next;
      x_curr = x_next;
    }

    LCS output_lcs = LCS(A, B, D, d, E, F, H, c, dt_);
    return std::make_pair(output_lcs, x_hat);

  }

  void iC3::UpdateQuaternionCosts(
    MatrixXd x_hat, const Eigen::VectorXd& x_des) {
    
    Q_.clear();
    R_.clear();
    G_.clear();
    U_.clear();


    double discount_factor = 1;
    for (int i = 0; i < N_; i++) {
      Q_.push_back(discount_factor * controller_options_.c3_options.Q);
      discount_factor *=  controller_options_.c3_options.gamma;
      if (i < N_) {
        R_.push_back(discount_factor * controller_options_.c3_options.R);
        G_.push_back(controller_options_.c3_options.G);
        U_.push_back(controller_options_.c3_options.U);
      }
    }  
    Q_.push_back(discount_factor * controller_options_.c3_options.Q); 

    for (int i = 0; i < N_ + 1; i++) {
      for (int index : controller_options_.quaternion_indices) {
      
        // make quaternion costs time-varying based on x_hat
        Eigen::VectorXd quat_curr_i = x_hat.col(i).segment(index, 4);
        Eigen::VectorXd quat_des_i = x_des.segment(index, 4);

        Eigen::MatrixXd quat_hessian_i = common::hessian_of_squared_quaternion_angle_difference(quat_curr_i, quat_des_i);

        // Regularize hessian so Q is always PSD
        double min_eigenval = quat_hessian_i.eigenvalues().real().minCoeff();
        Eigen::MatrixXd quat_regularizer_1 = std::max(0.0, -min_eigenval) * Eigen::MatrixXd::Identity(4, 4);
        Eigen::MatrixXd quat_regularizer_2 = quat_des_i * quat_des_i.transpose();
        Eigen::MatrixXd quat_regularizer_3 = 1e-8 * Eigen::MatrixXd::Identity(4, 4);

        double discount_factor = 1;
          Q_[i].block(index, index, 4, 4) = 
            discount_factor * controller_options_.quaternion_weight * 
            (quat_hessian_i + quat_regularizer_1 + 
            controller_options_.quaternion_regularizer_fraction * quat_regularizer_2 + quat_regularizer_3);
          discount_factor *= controller_options_.c3_options.gamma;

        // double q_min_eigenval = Q_[i].eigenvalues().real().minCoeff();
        // std::cout << "Q_" << i << " min eigenvalue " <<  q_min_eigenval << std::endl;
      }
    }
    Q_[N_] = Q_[N_-1];
  }

  
} // namespace systems
} // namespace c3