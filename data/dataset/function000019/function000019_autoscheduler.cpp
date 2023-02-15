#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000019");
	var i0("i0", 1, 1025), i1("i1", 0, 1024), i0_p1("i0_p1", 0, 1026);
	input icomp00("icomp00", {i0_p1}, p_float64);
	computation comp00("comp00", {i0,i1},  p_float64);
	comp00.set_expression(expr(0.0) - expr(0.890)*icomp00(i0)*icomp00(i0)*icomp00(i0 + 1) + icomp00(i0) - icomp00(i0 - 1) + 3.010);
	buffer buf00("buf00", {1026}, p_float64, a_output);
	icomp00.store_in(&buf00);
	comp00.store_in(&buf00, {i0});

	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf00}, "function000019.o", "./function000019_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, exec_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function000019_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}