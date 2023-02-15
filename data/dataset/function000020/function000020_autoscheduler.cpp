#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000020");
	var i0("i0", 0, 1024), i1("i1", 0, 256), i2("i2", 0, 256), i3("i3", 0, 2048), i4("i4", 0, 256), i5("i5", 1, 257), i5_p1("i5_p1", 0, 258);
	input icomp00("icomp00", {i0,i1}, p_float64);
	input icomp01("icomp01", {i1,i4,i5_p1}, p_float64);
	computation comp00("comp00", {i0,i1,i2,i3},  p_float64);
	comp00.set_expression(expr(2.000)*icomp00(i0, i1));
	computation comp01("comp01", {i0,i1,i4,i5},  p_float64);
	comp01.set_expression((expr(1.120)*icomp01(i1, i4, i5) - icomp01(i1, i4, i5 - 1))*icomp00(i0, i1) - icomp01(i1, i4, i5 + 1));
	comp00.then(comp01, i1);
	buffer buf00("buf00", {1024,256}, p_float64, a_output);
	buffer buf01("buf01", {256,256,258}, p_float64, a_output);
	icomp00.store_in(&buf00);
	icomp01.store_in(&buf01);
	comp00.store_in(&buf00, {i0,i1});
	comp01.store_in(&buf01, {i1,i4,i5});

	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf00,&buf01}, "function000020.o", "./function000020_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, exec_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function000020_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}