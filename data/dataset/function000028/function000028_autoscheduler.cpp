#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000028");
	var i0("i0", 0, 64), i1("i1", 0, 32), i2("i2", 0, 32), i3("i3", 0, 32), i4("i4", 0, 32), i5("i5", 0, 32), i6("i6", 0, 96);
	input icomp00("icomp00", {i0,i1,i2,i3}, p_float64);
	input input01("input01", {i2,i4}, p_float64);
	input input02("input02", {i0,i1,i4}, p_float64);
	input icomp01("icomp01", {i0,i1,i2,i3}, p_float64);
	input input04("input04", {i2}, p_float64);
	computation comp00("comp00", {i0,i1,i2,i3,i4,i5,i6},  p_float64);
	comp00.set_expression(expr(0.658)*icomp00(i0, i1, i2, i3) + input01(i2, i4) - input02(i0, i1, i4));
	computation comp01("comp01", {i0,i1,i2,i3,i4,i5,i6},  p_float64);
	comp01.set_expression(icomp01(i0, i1, i2, i3) + input04(i2)/icomp00(i0, i1, i2, i3));
	comp00.then(comp01, i6);
	buffer buf00("buf00", {64,32,32,32}, p_float64, a_output);
	buffer buf01("buf01", {32,32}, p_float64, a_input);
	buffer buf02("buf02", {64,32,32}, p_float64, a_input);
	buffer buf03("buf03", {64,32,32,32}, p_float64, a_output);
	buffer buf04("buf04", {32}, p_float64, a_input);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	input02.store_in(&buf02);
	icomp01.store_in(&buf03);
	input04.store_in(&buf04);
	comp00.store_in(&buf00, {i0,i1,i2,i3});
	comp01.store_in(&buf03, {i0,i1,i2,i3});

	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf00,&buf01,&buf02,&buf03,&buf04}, "function000028.o", "./function000028_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, exec_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function000028_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}