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
	tiramisu::codegen({&buf00,&buf01}, "function000020.o"); 
	return 0; 
}