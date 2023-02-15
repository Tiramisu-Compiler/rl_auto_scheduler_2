#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000026");
	var i0("i0", 0, 768), i1("i1", 0, 512), i2("i2", 0, 128), i3("i3", 0, 768), i4("i4", 0, 1280), i5("i5", 0, 128), i1_p1("i1_p1", 0, 513);
	input icomp00("icomp00", {i3,i4}, p_float64);
	input input01("input01", {i1_p1,i0}, p_float64);
	input icomp01("icomp01", {i0,i1}, p_float64);
	input input03("input03", {i1,i5}, p_float64);
	computation comp00("comp00", {i0,i1,i2,i3,i4},  p_float64);
	comp00.set_expression(expr(4.780)*icomp00(i3, i4)*input01(i1 + 1, i0));
	computation comp01("comp01", {i0,i1,i5},  p_float64);
	comp01.set_expression(icomp01(i0, i1) + input03(i1, i5));
	computation comp02("comp02", {i0,i1,i5}, icomp01(i0, i1) + expr(3.120)*input01(i1, i0));
	comp00.then(comp01, i1)
		.then(comp02, i5);
	buffer buf00("buf00", {768,1280}, p_float64, a_output);
	buffer buf01("buf01", {513,768}, p_float64, a_input);
	buffer buf02("buf02", {768,512}, p_float64, a_output);
	buffer buf03("buf03", {512,128}, p_float64, a_output);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	icomp01.store_in(&buf02);
	input03.store_in(&buf03);
	comp00.store_in(&buf00, {i3,i4});
	comp01.store_in(&buf02, {i0,i1});
	comp02.store_in(&buf03, {i1,i5});

	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf00,&buf01,&buf02,&buf03}, "function000026.o", "./function000026_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, exec_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function000026_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}