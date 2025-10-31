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
using std::vector;

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

  std::vector<MatrixXd> iC3::ComputeTrajectory(
    drake::systems::Context<double>& context,
    drake::systems::Context<drake::AutoDiffXd>& context_ad, 
    const std::vector<drake::SortedPair<drake::geometry::GeometryId>>& contact_geoms) {

    std::vector<double> x_init = *controller_options_.x_init;
    VectorXd x0 = Eigen::Map<VectorXd>(x_init.data(), x_init.size());    

    std::vector<double> x_des = *controller_options_.x_des;
    VectorXd xd = Eigen::Map<VectorXd>(x_des.data(), x_des.size());  

    MatrixXd x_hat = x0.transpose().replicate(N_, 1);
    std::vector<VectorXd> u_hat(N_, Eigen::VectorXd::Zero(n_u_));

    int num_iters = 50;

    // Set up time varying LCS
    std::vector<Eigen::MatrixXd> A;
    std::vector<Eigen::MatrixXd> B;
    std::vector<Eigen::MatrixXd> D;
    std::vector<Eigen::VectorXd> d;
    std::vector<Eigen::MatrixXd> E;
    std::vector<Eigen::MatrixXd> F;
    std::vector<Eigen::MatrixXd> H;
    std::vector<Eigen::VectorXd> c;

    std::vector<MatrixXd> all_x_hats;
    all_x_hats.push_back(x_hat);

    for (int iter = 0; iter < num_iters; iter++) {
      A.clear();
      B.clear();
      D.clear();
      d.clear();
      E.clear();
      F.clear();
      H.clear();
      c.clear();
      
      // Create time-varying LCS, linearized about x_hat
      for (int k = 0; k < N_; k++) {

        LCSFactory lcs_factory(plant_, context, plant_ad_, context_ad, 
            contact_geoms, controller_options_.lcs_factory_options);
      
        // Set context to current nominal state and input
        lcs_factory.UpdateStateAndInput(x_hat.row(k), u_hat[k]);

        LCS lcs = lcs_factory.GenerateLCS();

        A.push_back(lcs.A()[0]);
        B.push_back(lcs.B()[0]);
        D.push_back(lcs.D()[0]);
        d.push_back(lcs.d()[0]);
        E.push_back(lcs.E()[0]);
        F.push_back(lcs.F()[0]);
        H.push_back(lcs.H()[0]);
        c.push_back(lcs.c()[0]);
      }

      LCS time_varying_lcs = LCS(A, B, D, d, E, F, H, c, dt_);

      UpdateQuaternionCosts(x_hat, xd);
      C3::CostMatrices new_costs(Q_, R_, G_, U_);
      c3_->UpdateCostMatrices(new_costs);

      std::vector<VectorXd> target =
        std::vector<VectorXd>(N_ + 1, xd);
      c3_->UpdateLCS(time_varying_lcs);
      c3_->UpdateTarget(target);
      c3_->Solve(x_hat.row(0));

      std::vector<Eigen::VectorXd> x_sol = c3_->GetStateSolution();

      for (int k = 0; k < N_; k++) {
        x_hat.row(k) = x_sol[k].transpose();
      }
      all_x_hats.push_back(x_hat);
    }
    std::cout << "returned all_x_hats" << std::endl;
    return all_x_hats;
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
        Eigen::VectorXd quat_curr_i = x_hat.row(i).segment(index, 4);
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
      }
    }
  }

  
} // namespace systems
} // namespace c3