#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000031");
	var i0("i0", 0, 96), i1("i1", 1, 97), i2("i2", 0, 32), i3("i3", 0, 32), i4("i4", 0, 32), i5("i5", 0, 96), i1_p1("i1_p1", 0, 98), i1_p0("i1_p0", 0, 97);
	input icomp00("icomp00", {i0,i1_p0,i2}, p_float64);
	input icomp01("icomp01", {i0,i1_p0}, p_float64);
	input input02("input02", {i1_p1,i3}, p_float64);
	input input04("input04", {i1_p0,i5}, p_float64);
	computation comp00("comp00", {i0,i1,i2},  p_float64);
	comp00.set_expression(icomp00(i0, i1, i2)*icomp00(i0, i1 - 1, i2) - 5.050);
	computation comp01("comp01", {i0,i1,i3},  p_float64);
	comp01.set_expression(icomp01(i0, i1)*icomp01(i0, i1) + input02(i1 + 1, i3));
	computation comp02("comp02", {i0,i1,i4,i5}, expr(0.0) - 4.620 + input04(i1, i5)/icomp01(i0, i1));
	comp00.then(comp01, i1)
		.then(comp02, i1);
	buffer buf00("buf00", {96,97,32}, p_float64, a_output);
	buffer buf01("buf01", {96,97}, p_float64, a_output);
	buffer buf02("buf02", {98,32}, p_float64, a_input);
	buffer buf03("buf03", {96,97,32,96}, p_float64, a_output);
	buffer buf04("buf04", {97,96}, p_float64, a_input);
	icomp00.store_in(&buf00);
	icomp01.store_in(&buf01);
	input02.store_in(&buf02);
	input04.store_in(&buf04);
	comp00.store_in(&buf00);
	comp01.store_in(&buf01, {i0,i1});
	comp02.store_in(&buf03);

	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf00,&buf01,&buf02,&buf03,&buf04}, "function000031.o", "./function000031_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, exec_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function000031_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}