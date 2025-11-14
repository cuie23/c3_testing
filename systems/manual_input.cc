#include "systems/manual_input.h"

namespace c3 {
namespace systems {


ManualInput::ManualInput(const MultibodyPlant<double>& plant, int N, double dt) :
  plant_(plant),
  N_(N),
  dt_(dt) {

  // Declare output port for inputs.
  // Hardcoded to u=5 for plate-cube example
  output_port_index_ =
      this->DeclareVectorOutputPort("u", 5, &ManualInput::ComputeManualInput)
          .get_index();

}


void ManualInput::ComputeManualInput(const drake::systems::Context<double>& context,
              drake::systems::BasicVector<double>* output) const {
  double t = context.get_time();
  
  std::cout << "time: " << t << std::endl;

  std::unique_ptr<drake::systems::Context<double>> plant_context = plant_.CreateDefaultContext();		
	auto& context_ref = *plant_context;  
  VectorXd tau_g = plant_.CalcGravityGeneralizedForces(context_ref);
  Eigen::VectorXd u_gravity = Eigen::VectorXd::Zero(5);
  u_gravity[2] = -1 * (tau_g[2] + tau_g[10]); // Hard-coded cube + plate

  if (t > dt_ * N_ + 5.0 || t < 5.0) { 
    // Just compensate gravity if time is past horizon


    output->SetFromVector(u_gravity);
  

  } else {


    VectorXd u_sol(VectorXd::Zero(5));
    double k = (t - 5.0) / dt_;

    if (k < 10) {
      u_sol[2] = 80 - 120 / (1 + std::exp(-0.8 * (k-5))); // throw block
      
      u_sol[4] = u_sol[2];


      std::cout << u_sol[2] << std::endl;
      output->SetFromVector(u_sol);
    } else if (k < 17) {
      output->SetFromVector(u_gravity * 1.2);
    } else {
      output->SetFromVector(u_gravity);
    } 


  }
}


} // namespace systems
} // namespace c3