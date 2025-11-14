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
    C3ControllerOptions controller_options, iC3Options ic3_options)
    : plant_(plant),
      plant_ad_(plant_ad),
      controller_options_(controller_options),
      ic3_options_(ic3_options),
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

  vector<vector<MatrixXd>> iC3::ComputeTrajectory(
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
    MatrixXd c3_xs = MatrixXd::Zero(n_x_, N_);

    // Set initial guess to something kinda reasonable
    // TODO: make this a yaml option or use drake slerp
    VectorXd x_diff = xd - x0;
    for (int k = 0; k < N_+1; k++) {
      x_hat.col(k) = x0 + k * x_diff / (N_+1);

      if (k < N_) u_hat(2, k) = 10;
    }

    vector<MatrixXd> all_x_hats;
    vector<MatrixXd> all_u_hats;
    vector<MatrixXd> all_c3_x;

    all_x_hats.push_back(x_hat);
    all_u_hats.push_back(u_hat);

    LCSFactory lcs_factory(plant_, context, plant_ad_, context_ad, 
        contact_geoms, controller_options_.lcs_factory_options);

    LCS lcs = MakeTimeVaryingLCS(x_hat, u_hat, lcs_factory);

    vector<VectorXd> u_sol_for_penalization;

    vector<VectorXd> c3_quat_norms;
    for (int index : controller_options_.quaternion_indices) {
      c3_quat_norms.push_back(VectorXd::Ones(N_));
    }

    int num_iters = ic3_options_.num_iters;
    for (int iter = 0; iter < num_iters-1; iter++) {

      std::cout << "iC3 iteration " << iter << std::endl;
      UpdateQuaternionCosts(x_hat, xd, c3_quat_norms);

      MatrixXd Q_test = Q_[6];

      // MatrixXd cooked = (Q_[6] - Q_test);
      // std::cout << cooked << std::endl; 

      // Add actuation/position limits
      Eigen::MatrixXd A = Eigen::MatrixXd::Zero(23, 23);
      // A(0, 0) = 1;
      // A(1, 1) = 1;
      A(2, 2) = 1;
      A(3, 3) = 1;
      A(4, 4) = 1;
      Eigen::VectorXd lower_bound(Eigen::VectorXd::Zero(23));
      Eigen::VectorXd upper_bound(Eigen::VectorXd::Zero(23));

      // Plate position constraints
      lower_bound[0] = -0.2;
      lower_bound[1] = -0.2;
      lower_bound[2] = -1; 
      upper_bound[0] = 0.2;
      upper_bound[1] = 0.2;
      upper_bound[2] = 0.5;
      
      // Plate rotation constraints
      lower_bound[3] = -0.5;
      lower_bound[4] = -0.5;
      upper_bound[3] = 0.5;
      upper_bound[4] = 0.5;

      // Actuation limits
      Eigen::MatrixXd A_u = Eigen::MatrixXd::Zero(5, 5);
      A_u(0, 0) = 1;
      A_u(1, 1) = 1;
      A_u(2, 2) = 1;
      Eigen::VectorXd lower_bound_u(Eigen::VectorXd::Zero(5));
      Eigen::VectorXd upper_bound_u(Eigen::VectorXd::Zero(5));
      lower_bound_u << -20, -20, -20, 0, 0;
      upper_bound_u << 20, 20, 40, 0, 0; // plate + block is ~10 N 
      // c3_->AddLinearConstraint(A_u, lower_bound_u, upper_bound_u,
      //                                   ConstraintVariable::INPUT);
                 
      vector<Eigen::VectorXd> x_sol;
      vector<Eigen::VectorXd> u_sol;
        
      std::vector<VectorXd> target =
        std::vector<VectorXd>(N_ + 1, xd);


      int segment_length = N_ / ic3_options_.num_segments;
      VectorXd x_start = x_hat.col(0);
      int indexer = 0;
      for (int i = 0; i < ic3_options_.num_segments; i++) {
        LCS shortened_lcs = ShortenLCSFront(lcs, i * segment_length);
        C3::CostMatrices shortened_costs = ShortenCostsFront(i * segment_length);
        std::vector<VectorXd> shortened_targets;

        // Scale quaternions in target based on previous iteration
        for (int k = i * segment_length; k < N_+1; k++) {
          VectorXd target_k = target[k];
          for (int j = 0; j < controller_options_.quaternion_indices.size(); j++) {
              int index = controller_options_.quaternion_indices[j];
              double norm = c3_quat_norms[j](k);
              target_k.segment(index, 4) *= norm;
           }
           shortened_targets.push_back(target_k);
        }

        // Update c3_ to match new length
        c3_ = std::make_unique<C3Plus>(shortened_lcs, shortened_costs, shortened_targets,
                                 controller_options_.c3_options);

        if (i == 0 && iter == 0) {
            // On first iteration just use 0s
        } else if (i == 0) { 
          // Take from previous iC3 iteration
          vector<VectorXd> u_sol_keep = u_sol_for_penalization;
          c3_->set_u_sol(u_sol_keep);
          u_sol_for_penalization.clear();
        } else {
          vector<VectorXd> u_sol_keep(u_sol.begin() + segment_length, u_sol.end());
          c3_->set_u_sol(u_sol_keep);
        } 


        if (ic3_options_.add_position_constraints) {
          c3_->AddLinearConstraint(A, lower_bound, upper_bound,
                                            ConstraintVariable::STATE);
        }
                          
        c3_->Solve(x_start);

        x_sol = c3_->GetStateSolution();
        u_sol = c3_->GetInputSolution();

        // Only keep segment_length x's and u's
        for (int j = 0; j < segment_length; j++) {
          c3_xs.col(indexer) = x_sol[j];
          u_hat.col(indexer) = u_sol[j];
          indexer++;
          u_sol_for_penalization.push_back(u_sol[j]);
        }
        if (i < ic3_options_.num_segments - 1) {
          x_start = x_sol[segment_length];
        }
      }
      for (int i = indexer; i < N_; i++) {
        c3_xs.col(i) = x_sol[i - indexer];
        u_hat.col(i) = u_sol[i - indexer];
        u_sol_for_penalization.push_back(u_sol[i - indexer]);
      } 

      // Update norms of c3 quaternion outputs
      c3_quat_norms.clear();
      for (int index : controller_options_.quaternion_indices) {
        VectorXd c3_norm(N_+1);
        for (int i = 0; i < N_; i++) {
          c3_norm(i) = c3_xs.col(i).segment(index, 4).norm();
        } 
        c3_norm(N_) = c3_norm(N_-1); // c3 xsol only has N_ points
        c3_quat_norms.push_back(c3_norm);
        
      }

      auto output = DoLCSRollout(x0, u_hat, lcs_factory);
      lcs = output.first;
      x_hat = output.second;

      // normalize xhat quaternions
      // for (int i = 0; i < x_hat.cols(); i++) {
      //   for (int index : controller_options_.quaternion_indices) {
      //     x_hat.col(i).segment(index, 4).normalize();
      //   }
      // }

      all_x_hats.push_back(x_hat);
      all_u_hats.push_back(u_hat);
      all_c3_x.push_back(c3_xs);

      // Print costs
      
      if (ic3_options_.print_costs && iter % 3 == 2) {
        double x_cost = 0;
        double pos_cost = 0;
        double rot_cost = 0;

        vector<VectorXd> xds;
        for (int k = 0; k < N_+1; k++) {
          VectorXd target_k = target[k];
          for (int j = 0; j < controller_options_.quaternion_indices.size(); j++) {
              int index = controller_options_.quaternion_indices[j];
              double norm = c3_quat_norms[j](k);
              target_k.segment(index, 4) *= norm;
           }
           xds.push_back(target_k);
        }

        for (int i = 0; i < N_; i++) {
          VectorXd x_curr = c3_xs.col(i);
          VectorXd xd = xds[i];

          x_cost += (x_curr - xd).transpose() * Q_[i] * (x_curr - xd);
         // std::cout << "i: " << i << ", cost: " << (x_curr - xd).transpose() * Q_[i] * (x_curr - xd) << std::endl;

          rot_cost += (x_curr.segment(5, 4) - xd.segment(5, 4)).transpose() * 
              Q_[i].block(5, 5, 4, 4) * (x_curr.segment(5, 4) - xd.segment(5, 4));
          
          std::cout << "i: " << i << ", rot cost: " << (x_curr.segment(5, 4) - xd.segment(5, 4)).transpose() * 
              Q_[i].block(5, 5, 4, 4) * (x_curr.segment(5, 4) - xd.segment(5, 4)) << std::endl;


          Eigen::Quaterniond q_curr(x_curr(5), x_curr(6), x_curr(7), x_curr(8));
          Eigen::Quaterniond q_des(xd(5), xd(6), xd(7), xd(8));
          Eigen::AngleAxisd angle_axis(q_des * q_curr.inverse());
          double angle = angle_axis.angle();

          std::cout << "angle: " << angle << std::endl;


          pos_cost += (x_curr.segment(9, 3) - xd.segment(9, 3)).transpose() * 
              Q_[i].block(9, 9, 3, 3) * (x_curr.segment(9, 3) - xd.segment(9, 3));
          pos_cost += (x_curr.segment(0, 3) - xd.segment(0, 3)).transpose() * 
              Q_[i].block(0, 0, 3, 3) * (x_curr.segment(0, 3) - xd.segment(0, 3));

          std::cout << "i: " << i << ", cube pos cost: " << (x_curr.segment(9, 3) - xd.segment(9, 3)).transpose() * 
              Q_[i].block(9, 9, 3, 3) * (x_curr.segment(9, 3) - xd.segment(9, 3)) << std::endl;
          std::cout << "i: " << i << ", plate pos cost: " << (x_curr.segment(0, 3) - xd.segment(0, 3)).transpose() * 
              Q_[i].block(0, 0, 3, 3) * (x_curr.segment(0, 3) - xd.segment(0, 3)) << std::endl << std::endl;
        } 


        double u_cost = 0;
        VectorXd u_prev(VectorXd::Zero(5));
        for (int i = 0; i < R_.size(); i++) {
          VectorXd u_curr = u_hat.col(i);
          u_cost += (u_curr - u_prev).transpose() * R_[i] * (u_curr - u_prev);
        } 

        std::cout << "x cost: " << x_cost << std::endl;
        std::cout << "position cost: " << pos_cost << std::endl;
        std::cout << "rotation cost: " << rot_cost << std::endl;
        std::cout << "u cost: " << u_cost << std::endl;
      }
        
      

    }
    std::cout << "returned all_x_hats" << std::endl;
    vector<vector<MatrixXd>> outputs;
    outputs.push_back(all_x_hats);
    outputs.push_back(all_u_hats);
    outputs.push_back(all_c3_x);

    return outputs;
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

  LCS iC3::MakeTimeVaryingLCS(MatrixXd x_hat, MatrixXd u_hat, LCSFactory factory) {
    vector<Eigen::MatrixXd> A;
    vector<Eigen::MatrixXd> B;
    vector<Eigen::MatrixXd> D;
    vector<Eigen::VectorXd> d;
    vector<Eigen::MatrixXd> E;
    vector<Eigen::MatrixXd> F;
    vector<Eigen::MatrixXd> H;
    vector<Eigen::VectorXd> c;

    for (int k = 0; k < N_; k++) {
      
      // Linearize about kth xhat, uhat
      factory.UpdateStateAndInput(x_hat.col(k), u_hat.col(k));
      LCS lcs = factory.GenerateLCS();
      A.push_back(lcs.A()[0]);
      B.push_back(lcs.B()[0]);
      D.push_back(lcs.D()[0]);
      d.push_back(lcs.d()[0]);
      E.push_back(lcs.E()[0]);
      F.push_back(lcs.F()[0]);
      H.push_back(lcs.H()[0]);
      c.push_back(lcs.c()[0]);      
    }

    return LCS(A, B, D, d, E, F, H, c, dt_);
  }

  LCS iC3::ShortenLCSFront(LCS lcs, int num_timesteps_to_remove) {
    vector<Eigen::MatrixXd> A(lcs.A().begin() + num_timesteps_to_remove, lcs.A().end());
    vector<Eigen::MatrixXd> B(lcs.B().begin() + num_timesteps_to_remove, lcs.B().end());
    vector<Eigen::MatrixXd> D(lcs.D().begin() + num_timesteps_to_remove, lcs.D().end());
    vector<Eigen::VectorXd> d(lcs.d().begin() + num_timesteps_to_remove, lcs.d().end());
    vector<Eigen::MatrixXd> E(lcs.E().begin() + num_timesteps_to_remove, lcs.E().end());
    vector<Eigen::MatrixXd> F(lcs.F().begin() + num_timesteps_to_remove, lcs.F().end());
    vector<Eigen::MatrixXd> H(lcs.H().begin() + num_timesteps_to_remove, lcs.H().end());
    vector<Eigen::VectorXd> c(lcs.c().begin() + num_timesteps_to_remove, lcs.c().end());

    return LCS(A, B, D, d, E, F, H, c, dt_);
  }

  C3::CostMatrices iC3::ShortenCostsFront(int num_timesteps_to_remove) {
    vector<Eigen::MatrixXd> Q(Q_.begin() + num_timesteps_to_remove, Q_.end());
    vector<Eigen::MatrixXd> R(R_.begin() + num_timesteps_to_remove, R_.end());
    vector<Eigen::MatrixXd> G(G_.begin() + num_timesteps_to_remove, G_.end());
    vector<Eigen::MatrixXd> U(U_.begin() + num_timesteps_to_remove, U_.end());

    return C3::CostMatrices(Q, R, G, U);
  }


  void iC3::UpdateQuaternionCosts(
    MatrixXd x_hat, const Eigen::VectorXd& x_des, vector<VectorXd> c3_quat_norms) {
    
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
      int j = 0;
      for (int index : controller_options_.quaternion_indices) {
        double norm = c3_quat_norms[j](i);

        // make quaternion costs time-varying based on x_hat
        Eigen::VectorXd quat_curr_i = x_hat.col(i).segment(index, 4).normalized();
        Eigen::VectorXd quat_des_i = x_des.segment(index, 4);

        //std::cout << "xhat q: " << quat_curr_i.transpose() << std::endl;

        Eigen::MatrixXd quat_hessian_i = common::hessian_of_squared_quaternion_angle_difference(quat_curr_i, quat_des_i);

        // Regularize hessian so Q is always PSD
        double min_eigenval = quat_hessian_i.eigenvalues().real().minCoeff();
        //std::cout << min_eigenval << std::endl;

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
        j++;
      }
    }
    //std::cout << std::endl;
    //Q_[N_] = Q_[N_-1];
  }

  
} // namespace systems
} // namespace c3