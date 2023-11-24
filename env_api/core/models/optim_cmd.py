from env_api.scheduler.models.action import *


class OptimizationCommand:
    def __init__(self, action: Action, get_tiramisu_optim_str: bool = True):
        self.params_list = action.params
        self.action = action
        # A list of concerned computations of the actions
        self.comps = action.comps
        # We save the schedule of an action in each comp individually to form the whole schedule of a program later
        self.comps_schedule = {}
        if get_tiramisu_optim_str:
            self.tiramisu_optim_str = self.get_tiramisu_optim_str()

    def get_tiramisu_optim_str(self):
        """Convert the optimization command into Tiramisu code.
        Returns:
            str: The tiramisu snippet that represents the optimization command.
        """
        optim_str = ""
        if isinstance(self.action, Interchange):
            assert len(self.params_list) == 2
            interchange_str = (
                ".interchange(" + ",".join([str(p) for p in self.params_list]) + ");"
            )
            optim_str = ""
            for comp in self.comps:
                self.comps_schedule[comp] = "I(L{},L{})".format(*self.params_list)
                optim_str += "\n\t{}".format(comp) + interchange_str
            return optim_str
        elif isinstance(self.action, Skewing):
            assert len(self.params_list) == 4
            skewing_str = ".skew(" + ",".join([str(p) for p in self.params_list]) + ");"
            optim_str = ""
            for comp in self.comps:
                self.comps_schedule[comp] = "S(L{},L{},{},{})".format(*self.params_list)
                optim_str += "\n\t{}".format(comp) + skewing_str
            return optim_str

        elif isinstance(self.action, Parallelization):
            assert len(self.params_list) == 1
            for comp in self.comps:
                self.comps_schedule[comp] = "P(L{})".format(self.params_list[0])
            return (
                "\n\t"
                + self.comps[0]
                + ".tag_parallel_level("
                + str(self.params_list[0])
                + ");"
            )

        elif isinstance(self.action, Tiling):
            assert len(self.params_list) in [2, 4, 6]
            tiling_str = ".tile(" + ",".join([str(p) for p in self.params_list]) + ");"
            optim_str = ""
            for comp in self.comps:
                if len(self.params_list) == 2:
                    self.comps_schedule[comp] = "T1(L{},{})".format(*self.params_list)
                elif len(self.params_list) == 4:
                    self.comps_schedule[comp] = "T2(L{},L{},{},{})".format(
                        *self.params_list
                    )
                else:
                    self.comps_schedule[comp] = "T3(L{},L{},L{},{},{},{})".format(
                        *self.params_list
                    )
                optim_str += "\n\t{}".format(comp) + tiling_str

            for index, _ in enumerate(self.action.subtilings):
                sub_tile = self.action.subtilings[index]
                tiling_str = (
                    ".tile(" + ",".join([str(p) for p in sub_tile.params]) + ");"
                )
                for comp in sub_tile.comps:
                    if len(sub_tile.params) == 2:
                        self.comps_schedule[comp] = "T1(L{},{})".format(
                            *sub_tile.params
                        )
                    elif len(sub_tile.params) == 4:
                        self.comps_schedule[comp] = "T2(L{},L{},{},{})".format(
                            *sub_tile.params
                        )
                    else:
                        self.comps_schedule[comp] = "T3(L{},L{},L{},{},{},{})".format(
                            *sub_tile.params
                        )
                    optim_str += "\n\t{}".format(comp) + tiling_str

            return optim_str
        elif isinstance(self.action, Unrolling):
            optim_str = ""
            for comp in self.comps:
                self.comps_schedule[comp] = "U(L{},{})".format(*self.params_list)
                optim_str = (
                    f"\n\t{comp}.unroll({self.params_list[0]},{self.params_list[1]});"
                )
            # unrolling_str = (
            #     ".tag_unroll_level(" +
            #     ",".join([str(p) for p in self.params_list]) + ");")
            # optim_str += "\n\t{}".format(self.comps[0]) + unrolling_str
            return optim_str
        elif isinstance(self.action, Reversal):
            reversal_str = ".loop_reversal(" + str(self.params_list[0]) + ");"
            optim_str = ""
            for comp in self.comps:
                self.comps_schedule[comp] = "R(L{})".format(self.params_list[0])
                optim_str += "\n\t{}".format(comp) + reversal_str
            return optim_str
        elif isinstance(self.action, Fusion):
            fusion_level = len(self.params_list[0]["iterators"]) - 1
            self.fusion_str = f"F({{{self.params_list[0]['name']},{self.params_list[1]['name']}}},L{fusion_level})"
            annotations = self.action.annotations
            computation_to_fuse = self.params_list[1]["name"]
            computations_of_first_iterator = []

            def recursively_add_comps(iterator):
                computations_of_first_iterator.extend(
                    annotations["iterators"][iterator]["computations_list"]
                )
                for child in annotations["iterators"][iterator]["child_iterators"]:
                    recursively_add_comps(child)

            recursively_add_comps(self.params_list[0]["iterators"][-1])

            optim_str += f"""

    perform_full_dependency_analysis();
    {self.params_list[0]['name']}.then({self.params_list[1]['name']},{fusion_level});
    prepare_schedules_for_legality_checks(true);

    std::vector<std::tuple<tiramisu::var, int>> factors = tiramisu::global::get_implicit_function()->correcting_loop_fusion_with_shifting({{{", ".join([f"&{comp}" for comp in computations_of_first_iterator])}}}, {computation_to_fuse}, {{{", ".join([str(i) for i in range(len(self.params_list[0]['iterators']))])}}});
    for (const auto &tuple : factors)
    {{
        tiramisu::var var = std::get<0>(tuple);
        int value = std::get<1>(tuple);

        if (value != 0)
        {{
            {computation_to_fuse}.shift(var, value);
        }}
    }}"""
            # optim_str += f"{self.params_list[0]['name']}.then({self.params_list[1]['name']},{fusion_level});"
            return optim_str

    def __str__(self) -> str:
        return f"OptimizationCommand(action={self.action.__class__.__name__}, params={self.params_list})"

    def __repr__(self) -> str:
        return self.__str__()
