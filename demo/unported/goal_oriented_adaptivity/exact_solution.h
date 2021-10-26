#include <math.h>
#include <boost/math/constants/constants.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>

namespace py = pybind11;

const double pi_2 = boost::math::constants::pi<double>()/2.0;

class Exact : public dolfin::Expression
{
public:
  // Create expression with 1 component (scalar)
  Exact() : dolfin::Expression() {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const override
  {
    double r = sqrt(x[0]*x[0] + x[1]*x[1]);
    double theta = atan2(x[1], x[0]) + pi_2;
    values[0] = pow(r, 2.0/3.0)*sin((2.0/3.0)*theta);
  }
};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<Exact, std::shared_ptr<Exact>, dolfin::Expression>
    (m, "Exact")
    .def(py::init<>());
}
