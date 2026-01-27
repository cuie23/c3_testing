#include <string>
#include <vector>

#include <drake/common/yaml/yaml_io.h>

#include "common/find_resource.h"
#include "core/c3.h"
#include "core/c3_miqp.h"
#include "core/c3_qp.h"
#include "core/lcs.h"
#include "systems/c3_controller_options.h"
#include "systems/iC3_options.h"
#include "systems/framework/c3_output.h"
#include "systems/framework/timestamped_vector.h"

#include "drake/systems/analysis/simulator.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/leaf_system.h"

using std::vector;
using std::pair;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::multibody::MultibodyPlant;

namespace c3 {
namespace systems {

// For simulated rollouts
class InputSource : public drake::systems::LeafSystem<double> {
public:
  InputSource(MatrixXd u_hat, double dt, int N)  
  : u_hat_(u_hat),
    dt_(dt),
    N_(N) 
  {
    this->DeclareVectorOutputPort("u curr", u_hat_.rows(),
                                  &InputSource::CalcOutput);
  }

private:
  void CalcOutput(const Context<double>& context,
                  BasicVector<double>* output) const {
    double t = context.get_time();
    if (t < dt_ * N_) {
      int segment = (int)(t / dt_);
      output->get_mutable_value() = u_hat_.col(segment);
    } else {
      output->get_mutable_value() = VectorXd::Zero(u_hat_.rows());
    }
  }
  MatrixXd u_hat_;
  double dt_;
  int N_;
};

class iC3 : public drake::systems::LeafSystem<double> {

public:
 explicit iC3(
    drake::multibody::MultibodyPlant<double>& plant,
    drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    C3::CostMatrices& costs, 
    C3ControllerOptions controller_options, iC3Options ic3_options, 
    bool is_franka);

  vector<vector<MatrixXd>> ComputeTrajectory(
    drake::systems::Context<double>& context,
    drake::systems::Context<drake::AutoDiffXd>& context_ad, 
    const vector<drake::SortedPair<drake::geometry::GeometryId>>& contact_geoms);

private:
  
  // Given an initial x and u trajectory, return x rollout out using lcs
  pair<LCS, MatrixXd> DoLCSRollout(VectorXd x0, MatrixXd u_hat, LCSFactory factory);
  pair<LCS, MatrixXd> DoLCSRolloutLastIter(VectorXd x0, MatrixXd u_hat, LCS last_lcs, LCSFactory factory);
 
  MatrixXd RolloutUHat(VectorXd x0, MatrixXd u_hat);
  MatrixXd RolloutUHatFranka(VectorXd x0, MatrixXd u_hat);

  LCS MakeTimeVaryingLCS(MatrixXd x_hat, MatrixXd u_hat, LCSFactory factory);

  // removes num_timesteps_to_remove timesteps from the front of the LCS
  LCS ShortenLCSFront(LCS lcs, int num_timesteps_to_remove);


  C3::CostMatrices ShortenCostsFront(int num_timesteps_to_remove);

  // x_hat (N by n_x), kth row is x at time k
  void UpdateQuaternionCosts(
    MatrixXd x_hat, const Eigen::VectorXd& x_des, vector<VectorXd> c3_quat_norms);

  const drake::multibody::MultibodyPlant<double>& plant_;
  const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad_;

  // C3 options and solver configuration.
  C3ControllerOptions controller_options_;
  iC3Options ic3_options_;

  // Convenience variables for dimensions.
  int n_q_;       ///< Number of generalized positions.
  int n_v_;       ///< Number of generalized velocities.
  int n_x_;       ///< Total state dimension.
  int n_lambda_;  ///< Number of Lagrange multipliers.
  int n_u_;       ///< Number of control inputs.
  double dt_;     ///< Time step.

  // C3 solver instance.
  mutable std::unique_ptr<C3> c3_;

  // Cost matrices for optimization.
  mutable std::vector<Eigen::MatrixXd> Q_;  ///< State cost matrices.
  mutable std::vector<Eigen::MatrixXd> R_;  ///< Input cost matrices.
  mutable std::vector<Eigen::MatrixXd> G_;  ///< State-input cross-term matrices.
  mutable std::vector<Eigen::MatrixXd> U_;  ///< Constraint matrices.

  int N_;  ///< Horizon length.
  bool is_franka_;

};

} // namespace systems
} // namespace c3