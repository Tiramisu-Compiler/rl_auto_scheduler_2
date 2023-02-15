#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000034");
	var i0("i0", 1, 257), i1("i1", 1, 257), i2("i2", 1, 257), i0_p0("i0_p0", 0, 257), i1_p0("i1_p0", 0, 257), i0_p1("i0_p1", 0, 258), i1_p1("i1_p1", 0, 258), i2_p1("i2_p1", 0, 258);
	input icomp00("icomp00", {i0_p0,i1_p0}, p_float64);
	input input01("input01", {i0_p1,i1_p1,i2_p1}, p_float64);
	computation comp00("comp00", {i0,i1,i2},  p_float64);
	comp00.set_expression(((icomp00(i0, i1)*input01(i0, i1, i2) + input01(i0, i1, i2 + 1)*input01(i0 + 1, i1, i2))*input01(i0, i1, i2 - 1)*input01(i0, i1 + 1, i2) - input01(i0, i1 - 1, i2)*input01(i0 - 1, i1, i2)*input01(i0 + 1, i1, i2))/input01(i0 + 1, i1, i2));
	buffer buf00("buf00", {257,257}, p_float64, a_output);
	buffer buf01("buf01", {258,258,258}, p_float64, a_input);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	comp00.store_in(&buf00, {i0,i1});

	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf00,&buf01}, "function000034.o", "./function000034_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, exec_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function000034_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}