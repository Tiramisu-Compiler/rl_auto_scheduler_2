#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000022");
	var i0("i0", 1, 65), i1("i1", 1, 65), i0_p0("i0_p0", 0, 65), i0_p1("i0_p1", 0, 66), i1_p1("i1_p1", 0, 66), i1_p0("i1_p0", 0, 65);
	input icomp00("icomp00", {i0_p0,i1_p0}, p_float64);
	input input01("input01", {i1_p0}, p_float64);
	input input02("input02", {i0_p0}, p_float64);
	input input03("input03", {i0_p1,i1_p1}, p_float64);
	input input04("input04", {i1_p0}, p_float64);
	computation comp00("comp00", {i0,i1},  p_float64);
	comp00.set_expression((input03(i0, i1 - 1) + 3.810)*(input04(i1) + 5.470) + icomp00(i0, i1) - input01(i1) + expr(0.862)*input02(i0) - 2.720);
	computation comp01("comp01", {i0,i1}, expr(0.0) - icomp00(i0, i1)*input03(i0, i1 + 1) - input03(i0, i1 - 1)*input03(i0, i1 + 1) + input03(i0, i1 + 1)*input03(i0 + 1, i1 - 1) - input03(i0, i1 + 1)*input03(i0 - 1, i1 + 1)/input03(i0 - 1, i1 - 1) + input03(i0, i1 + 1)*input03(i0 + 1, i1)/(input03(i0 - 1, i1)*input03(i0 + 1, i1 + 1)));
	comp00.then(comp01, i1);
	buffer buf00("buf00", {65,65}, p_float64, a_output);
	buffer buf01("buf01", {65}, p_float64, a_input);
	buffer buf02("buf02", {65}, p_float64, a_input);
	buffer buf03("buf03", {66,66}, p_float64, a_output);
	buffer buf04("buf04", {65}, p_float64, a_input);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	input02.store_in(&buf02);
	input03.store_in(&buf03);
	input04.store_in(&buf04);
	comp00.store_in(&buf00);
	comp01.store_in(&buf03);

	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf00,&buf01,&buf02,&buf03,&buf04}, "function000022.o", "./function000022_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, exec_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function000022_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}